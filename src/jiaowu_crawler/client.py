from __future__ import annotations

import json
import re
import csv
import time
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Page, sync_playwright

from .config import Settings


@dataclass
class CrawlResult:
    raw: dict[str, Any]
    flat_rows: list[dict[str, Any]]


@dataclass
class NetworkRecord:
    method: str
    url: str
    status: int
    content_type: str
    request_body: str | None
    response_preview: str


@dataclass
class BjPdfResult:
    class_name: str
    pdf_path: Path
    selected_item: dict[str, Any]


@dataclass
class BjBatchResult:
    total_items: int
    exported_paths: list[Path]


@dataclass
class BjMultiGradeResult:
    grade_stats: dict[str, dict[str, int]]


def _is_login_url(url: str) -> bool:
    return "login_slogin" in (url or "")


def _extract_csrf_token(html: str) -> str | None:
    patterns = [
        r'name="csrftoken"\s+value="([^"]+)"',
        r'id="csrftoken"\s+value="([^"]+)"',
        r'var\s+csrftoken\s*=\s*"([^"]+)"',
    ]
    for pattern in patterns:
        m = re.search(pattern, html, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _flatten_records(raw: dict[str, Any]) -> list[dict[str, Any]]:
    kb_list = raw.get("kbList") or raw.get("result") or []
    if not isinstance(kb_list, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in kb_list:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "course": item.get("kcmc") or item.get("courseName"),
                "teacher": item.get("xm") or item.get("jsxm"),
                "weekday": item.get("xqjmc") or item.get("xqj"),
                "time": item.get("jcs") or item.get("jc"),
                "weeks": item.get("zcd"),
                "location": item.get("cdmc") or item.get("classroom"),
                "class": item.get("jxbmc") or item.get("teachingClass"),
                "raw": json.dumps(item, ensure_ascii=False),
            }
        )
    return rows


def _build_headers(settings: Settings, csrf: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
        "Origin": settings.base_url,
        "Referer": f"{settings.base_url}/jwglxt/kbcx/xskbcx_cxXsKb.html?gnmkdm=N2151",
        "X-Requested-With": "XMLHttpRequest",
        "User-Agent": "Mozilla/5.0",
    }
    if csrf:
        headers["csrftoken"] = csrf
    return headers


def _fetch_timetable(page: Page, settings: Settings, xnm: str, xqm: str, csrf: str | None) -> dict[str, Any]:
    candidate_endpoints = [
        "/jwglxt/kbcx/xskbcx_cxXsKb.html?gnmkdm=N2151",
        "/jwglxt/kbcx/xskbcx_cxXsKb.html",
        "/jwglxt/kbcx/xskbcx_cxXsKbxx.html?gnmkdm=N2151",
    ]
    payload = {
        "xnm": xnm,
        "xqm": xqm,
        "kzlx": "ck",
    }

    headers = _build_headers(settings, csrf)

    last_error: str | None = None
    for ep in candidate_endpoints:
        url = f"{settings.base_url}{ep}"
        resp = page.request.post(url, form=payload, headers=headers)
        if resp.ok:
            try:
                data = resp.json()
                if isinstance(data, dict):
                    if data.get("kbList") or data.get("result"):
                        return data
                    if data.get("msg") and data.get("status") == "error":
                        last_error = f"{url}: {data.get('msg')}"
                        continue
                    return data
            except Exception as ex:
                last_error = f"{url}: JSON parse failed: {ex}"
                continue
        else:
            last_error = f"{url}: HTTP {resp.status}"

    raise RuntimeError(last_error or "All timetable endpoints failed")


def _ensure_login(
    page: Page,
    settings: Settings,
    username: str | None = None,
    password: str | None = None,
) -> None:
    login_url = f"{settings.base_url}/xtgl/login_slogin.html"
    page.goto(login_url, wait_until="domcontentloaded", timeout=settings.timeout_ms)
    print("[INFO] 浏览器已打开，请手动完成登录。")
    if username and password:
        try:
            page.locator("#yhm").fill(username)
            page.locator("#mm").fill(password)
            page.locator("#dl").click()
            page.wait_for_timeout(2000)
            print("[INFO] 已尝试自动提交账号密码。")
        except Exception:
            print("[WARN] 自动填写账号密码失败，将转为手动登录。")

    for _ in range(3):
        try:
            page.goto(f"{settings.base_url}/xtgl/index_initMenu.html", wait_until="domcontentloaded", timeout=settings.timeout_ms)
        except Exception:
            page.wait_for_timeout(1000)
        if not _is_login_url(page.url):
            print("[INFO] 登录态校验通过。")
            return
        print("[INFO] 仍在登录页（可能需验证码/二次认证），请在浏览器完成登录后回车继续...")
        input()

    raise RuntimeError("登录未成功：系统仍停留在登录页，请检查账号密码、验证码或统一认证流程。")


def _parse_form_keys(form_body: str | None) -> list[str]:
    if not form_body:
        return []
    parsed = parse_qs(form_body, keep_blank_values=True)
    return sorted(parsed.keys())


def _save_network_summary(records: list[NetworkRecord], output_dir: Path) -> None:
    grouped: dict[str, dict[str, Any]] = {}

    for r in records:
        u = urlparse(r.url)
        path = u.path
        query_keys = sorted(parse_qs(u.query, keep_blank_values=True).keys())
        body_keys = _parse_form_keys(r.request_body)

        if path not in grouped:
            grouped[path] = {
                "methods": set(),
                "statuses": set(),
                "query_keys": set(),
                "body_keys": set(),
                "count": 0,
            }

        g = grouped[path]
        g["methods"].add(r.method)
        g["statuses"].add(str(r.status))
        g["query_keys"].update(query_keys)
        g["body_keys"].update(body_keys)
        g["count"] += 1

    lines: list[str] = ["# Network Trace Summary", ""]
    for path, g in sorted(grouped.items(), key=lambda x: x[0]):
        lines.append(f"## {path}")
        lines.append(f"- count: {g['count']}")
        lines.append(f"- methods: {', '.join(sorted(g['methods']))}")
        lines.append(f"- statuses: {', '.join(sorted(g['statuses']))}")
        lines.append(f"- query_keys: {', '.join(sorted(g['query_keys'])) or '(none)'}")
        lines.append(f"- body_keys: {', '.join(sorted(g['body_keys'])) or '(none)'}")
        lines.append("")

    (output_dir / "network_trace_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_flat_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = ["course", "teacher", "weekday", "time", "weeks", "location", "class", "raw"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _post_json(page: Page, url: str, form: dict[str, Any]) -> Any:
    resp = page.request.post(url, form=form)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status} for {url}")
    try:
        return resp.json()
    except Exception:
        return resp.text()


def _first_match(items: list[dict[str, Any]], class_name: str) -> dict[str, Any]:
    key = class_name.strip().lower()
    for item in items:
        candidates = [
            str(item.get("bjmc", "")),
            str(item.get("bj", "")),
            str(item.get("bh", "")),
            str(item.get("tjkbmc", "")),
        ]
        if any(key == c.strip().lower() for c in candidates if c):
            return item
    for item in items:
        candidates = [
            str(item.get("bjmc", "")),
            str(item.get("bj", "")),
            str(item.get("bh", "")),
            str(item.get("tjkbmc", "")),
        ]
        if any(key in c.strip().lower() for c in candidates if c):
            return item
    samples: list[str] = []
    for item in items[:10]:
        samples.append(
            str(item.get("bjmc") or item.get("bj") or item.get("bh") or item.get("tjkbmc") or "").strip()
        )
    raise RuntimeError(f"未找到班级: {class_name}。可用示例: {', '.join([s for s in samples if s])}")


def _xszd_map(page: Page, settings: Settings, gnmkdm: str) -> dict[str, str]:
    url = f"{settings.base_url}/kbdy/bjkbdy_cxKbzdxsxx.html?gnmkdm={gnmkdm}"
    data = _post_json(page, url, {"kbzl": "bj", "doType": "query"})
    result: dict[str, str] = {}
    if isinstance(data, list):
        for row in data:
            zdm = str(row.get("ZDM", "")).strip()
            sfxs = str(row.get("SFXS", "0")).strip()
            if zdm:
                result[f"xszd.{zdm}"] = "true" if sfxs == "1" else "false"
    return result


def _build_pdf_request(item: dict[str, Any], xszd: dict[str, str]) -> dict[str, Any]:
    req: dict[str, Any] = {}
    req.update(xszd)
    req["xnm"] = item.get("xnm", "")
    req["xqm"] = item.get("xqm", "")
    req["xnmc"] = item.get("xnmc", "")
    req["xqmmc"] = item.get("xqmmc", "")
    req["xqh_id"] = item.get("xqh_id", "")
    req["modelList[0].tjkbzdm"] = item.get("tjkbzdm", "")
    req["modelList[0].tjkbzxsdm"] = item.get("tjkbzxsdm", "")
    req["modelList[0].xnm"] = item.get("xnm", "")
    req["modelList[0].xqm"] = item.get("xqm", "")
    req["modelList[0].xnmc"] = item.get("xnmc", "")
    req["modelList[0].xqmmc"] = item.get("xqmmc", "")
    req["modelList[0].njdm_id"] = item.get("njdm_id", "")
    req["modelList[0].zyh_id"] = item.get("zyh_id", "")
    req["modelList[0].bh_id"] = item.get("bh_id", "")
    req["modelList[0].tjkbmc"] = item.get("tjkbmc") or item.get("bjmc") or item.get("bj") or ""
    req["modelList[0].xqh_id"] = item.get("xqh_id", "")
    req["modelList[0].zymc"] = item.get("zymc", "")
    req["modelList[0].jgmc"] = item.get("jgmc", "")
    req["modelList[0].bjmc"] = item.get("bj", "") or item.get("bjmc", "")
    req["modelList[0].xkrs"] = item.get("xkrs", "")
    req["modelList[0].jsxm"] = item.get("jsxm", "")
    req["modelList[0].lxdh"] = item.get("lxdh", "")
    req["modelList[0].bh"] = item.get("bh", "")
    req["modelList[0].njmc"] = item.get("njmc", "")
    req["modelList[0].zs"] = ""
    req["modelList[0].xsdm"] = ""
    req["modelList[0].zxszjjs"] = "false"
    req["modelList[0].kclxdm"] = ""
    req["modelList[0].kclbdm"] = ""
    req["modelList[0].kbsjlyqz"] = ""
    req["modelList[0].yf"] = ""
    return req


def _save_excel_response(
    page: Page,
    settings: Settings,
    output_dir: Path,
    class_name: str,
    xnm: str,
    xqm: str,
    resp: Any,
) -> Path:
    content_type = (resp.headers.get("content-type", "") if resp.headers else "").lower()
    safe_name = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff-]+", "_", class_name).strip("_") or "class"

    def _default_xls_path() -> Path:
        path = output_dir / f"{safe_name}_{xnm}_{xqm}.xls"
        if not path.exists():
            return path
        i = 2
        while True:
            p = output_dir / f"{safe_name}_{xnm}_{xqm}_{i}.xls"
            if not p.exists():
                return p
            i += 1

    if any(k in content_type for k in ["excel", "spreadsheet", "octet-stream"]):
        out = _default_xls_path()
        out.write_bytes(resp.body())
        return out

    txt = resp.text()
    if "success" in txt.lower():
        url = txt.split("#")[0].strip().strip('"').strip("'")
        if url:
            if url.startswith("/"):
                url = f"{settings.base_url}{url}"
            file_resp = page.request.get(url)
            if not file_resp.ok:
                raise RuntimeError(f"Excel 文件地址请求失败: HTTP {file_resp.status}, url={url}")
            out = _default_xls_path()
            out.write_bytes(file_resp.body())
            return out

    debug_path = output_dir / "excel_error_response.html"
    debug_path.write_text(txt, encoding="utf-8")
    raise RuntimeError(f"未返回 Excel，错误页已保存: {debug_path}")


def _query_bjkb_list(
    page: Page,
    settings: Settings,
    gnmkdm: str,
    xnm: str,
    xqm: str,
    xqh_id: str,
    njdm_id: str,
) -> dict[str, Any]:
    list_url = f"{settings.base_url}/kbdy/bjkbdy_cxBjkbdyTjkbList.html?gnmkdm={gnmkdm}"
    list_payload: dict[str, Any] = {
        "xnm": xnm,
        "xqm": xqm,
        "xqh_id": xqh_id,
        "njdm_id": njdm_id,
        "jg_id": "",
        "zyh_id": "",
        "zyfx_id": "",
        "bh_id": "",
        "xsdm": "",
        "pyccdm": "",
        "kclxdm": "",
        "kclbdm": "",
        "sfzhsjk": "",
        "kbsjlyqz": "",
        "zs": "",
        "yf": "",
        "_search": "false",
        "nd": str(int(time.time() * 1000)),
        "queryModel.showCount": "1000",
        "queryModel.currentPage": "1",
        "queryModel.sortName": "+",
        "queryModel.sortOrder": "asc",
        "time": "0",
    }
    data = _post_json(page, list_url, list_payload)
    if isinstance(data, dict):
        return data
    return {"_raw": data}


def _query_bjkb_list_page(
    page: Page,
    settings: Settings,
    gnmkdm: str,
    xnm: str,
    xqm: str,
    current_page: int,
    show_count: int,
    xqh_id: str = "",
    njdm_id: str = "",
) -> dict[str, Any]:
    list_url = f"{settings.base_url}/kbdy/bjkbdy_cxBjkbdyTjkbList.html?gnmkdm={gnmkdm}"
    payload: dict[str, Any] = {
        "xnm": xnm,
        "xqm": xqm,
        "xqh_id": xqh_id,
        "njdm_id": njdm_id,
        "jg_id": "",
        "zyh_id": "",
        "zyfx_id": "",
        "bh_id": "",
        "xsdm": "",
        "pyccdm": "",
        "kclxdm": "",
        "kclbdm": "",
        "sfzhsjk": "",
        "kbsjlyqz": "",
        "zs": "",
        "yf": "",
        "_search": "false",
        "nd": str(int(time.time() * 1000)),
        "queryModel.showCount": str(show_count),
        "queryModel.currentPage": str(current_page),
        "queryModel.sortName": "+",
        "queryModel.sortOrder": "asc",
        "time": "0",
    }
    data = _post_json(page, list_url, payload)
    if isinstance(data, dict):
        return data
    return {"_raw": data}


def _query_bjkb_list_via_ui(
    page: Page,
    settings: Settings,
    gnmkdm: str,
    xnm: str,
    xqm: str,
) -> dict[str, Any]:
    index_url = f"{settings.base_url}/kbdy/bjkbdy_cxBjkbdyIndex.html?gnmkdm={gnmkdm}&layout=default"
    page.goto(index_url, wait_until="domcontentloaded", timeout=settings.timeout_ms)
    page.wait_for_timeout(1500)

    data: Any = {"items": []}
    last_err = None
    for _ in range(3):
        try:
            # 按页面联动要求将筛选置为“全部”
            page.evaluate(
                """(args) => {
                    const setEmpty = (id) => {
                      if (!window.$) return false;
                      const el = window.$('#' + id);
                      if (!el || !el.length) return false;
                      el.val('');
                      el.trigger('change');
                      return true;
                    };
                    setEmpty('jg_id');
                    setEmpty('zyh_id');
                    setEmpty('zyfx_id');
                    setEmpty('bh_id');
                    if (window.$('#xnm').length) { window.$('#xnm').val(String(args.xnm)); window.$('#xnm').trigger('change'); }
                    if (window.$('#xqm').length) { window.$('#xqm').val(String(args.xqm)); window.$('#xqm').trigger('change'); }
                }""",
                {"xnm": xnm, "xqm": xqm},
            )
            page.wait_for_timeout(1200)

            with page.expect_response(
                lambda r: "bjkbdy_cxBjkbdyTjkbList.html" in r.url and r.request.method.upper() == "POST",
                timeout=settings.timeout_ms,
            ) as resp_info:
                page.evaluate(
                    """() => {
                        if (typeof searchResult === 'function') {
                          searchResult();
                          return;
                        }
                        const btns = Array.from(document.querySelectorAll('button,input[type="button"],a'));
                        const hit = btns.find(el => ((el.textContent || el.value || '').trim().includes('查询')));
                        if (hit) hit.click();
                    }"""
                )
            resp = resp_info.value
            try:
                data = resp.json()
            except Exception:
                data = {"_raw": resp.text()}
            if isinstance(data, dict) and isinstance(data.get("items"), list):
                break
        except Exception as ex:
            last_err = ex
            page.wait_for_timeout(1000)

    if last_err and not (isinstance(data, dict) and isinstance(data.get("items"), list)):
        raise RuntimeError(f"前端查询触发失败: {last_err}")
    if isinstance(data, dict):
        return data
    return {"_raw": data}


def _query_bjkb_list_via_ui_with_payload(
    page: Page,
    settings: Settings,
    gnmkdm: str,
    xnm: str,
    xqm: str,
    njdm_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    index_url = f"{settings.base_url}/kbdy/bjkbdy_cxBjkbdyIndex.html?gnmkdm={gnmkdm}&layout=default"
    page.goto(index_url, wait_until="domcontentloaded", timeout=settings.timeout_ms)
    page.wait_for_timeout(1500)

    with page.expect_response(
        lambda r: "bjkbdy_cxBjkbdyTjkbList.html" in r.url and r.request.method.upper() == "POST",
        timeout=settings.timeout_ms,
    ) as resp_info:
        page.evaluate(
            """(args) => {
                const setEmpty = (id) => {
                  if (!window.$) return false;
                  const el = window.$('#' + id);
                  if (!el || !el.length) return false;
                  el.val('');
                  el.trigger('change');
                  return true;
                };
                setEmpty('jg_id');
                if (args.njdm_id && window.$('#njdm_id').length) {
                  window.$('#njdm_id').val(String(args.njdm_id));
                  window.$('#njdm_id').trigger('change');
                }
                setEmpty('zyh_id');
                setEmpty('zyfx_id');
                setEmpty('bh_id');
                if (window.$('#xnm').length) { window.$('#xnm').val(String(args.xnm)); window.$('#xnm').trigger('change'); }
                if (window.$('#xqm').length) { window.$('#xqm').val(String(args.xqm)); window.$('#xqm').trigger('change'); }
                if (typeof searchResult === 'function') {
                  searchResult();
                  return;
                }
                const btns = Array.from(document.querySelectorAll('button,input[type="button"],a'));
                const hit = btns.find(el => ((el.textContent || el.value || '').trim().includes('查询')));
                if (hit) hit.click();
            }""",
            {"xnm": xnm, "xqm": xqm, "njdm_id": njdm_id or ""},
        )
    resp = resp_info.value
    try:
        data = resp.json()
    except Exception:
        data = {"_raw": resp.text()}

    post_data = resp.request.post_data or ""
    parsed = parse_qs(post_data, keep_blank_values=True)
    payload: dict[str, str] = {k: (v[0] if v else "") for k, v in parsed.items()}
    if "xnm" not in payload:
        payload["xnm"] = str(xnm)
    if "xqm" not in payload:
        payload["xqm"] = str(xqm)
    return (data if isinstance(data, dict) else {"_raw": data}, payload)


def run_bjkbdy_pdf(
    settings: Settings,
    output_dir: Path,
    class_name: str,
    xnm: str,
    xqm: str,
    njdm_id: str,
    xqh_id: str,
    gnmkdm: str = "N214505",
    username: str | None = None,
    password: str | None = None,
    headless: bool = False,
) -> BjPdfResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(accept_downloads=True, ignore_https_errors=True)
        page = context.new_page()
        _ensure_login(page, settings, username=username, password=password)

        index_url = f"{settings.base_url}/kbdy/bjkbdy_cxBjkbdyIndex.html?gnmkdm={gnmkdm}&layout=default"
        page.goto(index_url, wait_until="domcontentloaded", timeout=settings.timeout_ms)

        # 优先按页面真实交互流程查询（先把学院/专业/专业方向/班级设为全部）
        list_data = _query_bjkb_list_via_ui(
            page=page,
            settings=settings,
            gnmkdm=gnmkdm,
            xnm=xnm,
            xqm=xqm,
        )
        items: list[dict[str, Any]] = []
        if isinstance(list_data, dict):
            ui_items = list_data.get("items")
            if isinstance(ui_items, list):
                items = ui_items

        trials = [
            {"xqh_id": xqh_id, "njdm_id": njdm_id, "label": "xqh_id + njdm_id"},
            {"xqh_id": xqh_id, "njdm_id": "", "label": "only xqh_id"},
            {"xqh_id": "", "njdm_id": njdm_id, "label": "only njdm_id"},
            {"xqh_id": "", "njdm_id": "", "label": "no xqh_id/njdm_id"},
        ]

        used_trial = "ui_all_filters"
        if not items:
            for t in trials:
                list_data = _query_bjkb_list(
                    page=page,
                    settings=settings,
                    gnmkdm=gnmkdm,
                    xnm=xnm,
                    xqm=xqm,
                    xqh_id=t["xqh_id"],
                    njdm_id=t["njdm_id"],
                )
                current_items = list_data.get("items") if isinstance(list_data, dict) else None
                if isinstance(current_items, list) and current_items:
                    items = current_items
                    used_trial = t["label"]
                    break

        (output_dir / "bjkbdy_list_response.json").write_text(
            json.dumps(list_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not items:
            raise RuntimeError(
                "班级列表为空（已自动放宽筛选重试）。请检查 output/bjkbdy_list_response.json，"
                "并确认当前账号在该学年学期有班级课表权限。"
            )
        print(f"[INFO] 班级列表命中策略: {used_trial}，共 {len(items)} 条")

        selected = _first_match(items, class_name)
        (output_dir / "bjkbdy_selected_item.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        kz_url = f"{settings.base_url}/kbdy/bjkbdy_cxXnxqsfkz.html"
        kz_data = _post_json(page, kz_url, {"kzlx": "dy", "xnm": xnm, "xqm": xqm})
        if isinstance(kz_data, str) and "true" in kz_data.lower():
            raise RuntimeError("当前学年学期疑似已控制打印（cxXnxqsfkz 返回 true）")

        xszd = _xszd_map(page, settings, gnmkdm)
        request_map = _build_pdf_request(selected, xszd)

        excel_url = f"{settings.base_url}/kbdy/bjkbdy_cxDcExcelBjkb.html?doType=one"
        excel_resp = page.request.post(excel_url, form=request_map)
        if not excel_resp.ok:
            raise RuntimeError(f"Excel 请求失败: HTTP {excel_resp.status}")
        pdf_path = _save_excel_response(page, settings, output_dir, class_name, xnm, xqm, excel_resp)

        browser.close()

    return BjPdfResult(class_name=class_name, pdf_path=pdf_path, selected_item=selected)


def run_bjkbdy_export_all(
    settings: Settings,
    output_dir: Path,
    xnm: str,
    xqm: str,
    gnmkdm: str = "N214505",
    njdm_id: str | None = None,
    username: str | None = None,
    password: str | None = None,
    headless: bool = False,
) -> BjBatchResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_paths: list[Path] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(accept_downloads=True, ignore_https_errors=True)
        page = context.new_page()
        _ensure_login(page, settings, username=username, password=password)

        debug_lines: list[str] = []
        debug_lines.append(f"xnm={xnm}, xqm={xqm}, gnmkdm={gnmkdm}, njdm_id={njdm_id or ''}")

        all_items: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        first_data, base_payload = _query_bjkb_list_via_ui_with_payload(
            page=page, settings=settings, gnmkdm=gnmkdm, xnm=xnm, xqm=xqm, njdm_id=njdm_id
        )
        base_payload["queryModel.showCount"] = "150"
        base_payload["queryModel.currentPage"] = "1"
        base_payload["nd"] = str(int(time.time() * 1000))
        base_payload["time"] = "0"
        debug_lines.append("ui_payload=" + json.dumps(base_payload, ensure_ascii=False))

        first_data = _post_json(
            page,
            f"{settings.base_url}/kbdy/bjkbdy_cxBjkbdyTjkbList.html?gnmkdm={gnmkdm}",
            base_payload,
        )
        first_items = first_data.get("items") if isinstance(first_data, dict) else None
        if isinstance(first_items, list):
            for item in first_items:
                key = str(item.get("id") or item.get("bh_id") or item.get("tjkbzdm") or "")
                if key and key in seen_ids:
                    continue
                if key:
                    seen_ids.add(key)
                all_items.append(item)
        debug_lines.append(f"page1_items={len(first_items) if isinstance(first_items, list) else 0}")

        list_url = f"{settings.base_url}/kbdy/bjkbdy_cxBjkbdyTjkbList.html?gnmkdm={gnmkdm}"
        max_pages = 50
        for current_page in range(2, max_pages + 1):
            payload = copy.deepcopy(base_payload)
            payload["queryModel.showCount"] = "150"
            payload["queryModel.currentPage"] = str(current_page)
            payload["nd"] = str(int(time.time() * 1000))
            payload["time"] = "0"
            data = _post_json(page, list_url, payload)
            items = data.get("items") if isinstance(data, dict) else None
            if not isinstance(items, list) or not items:
                debug_lines.append(f"page{current_page}_items=0")
                break
            debug_lines.append(f"page{current_page}_items={len(items)}")
            for item in items:
                key = str(item.get("id") or item.get("bh_id") or item.get("tjkbzdm") or "")
                if key and key in seen_ids:
                    continue
                if key:
                    seen_ids.add(key)
                all_items.append(item)
            if len(items) < 150:
                break

        (output_dir / "bjkbdy_all_items.json").write_text(
            json.dumps(all_items, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "bjkbdy_export_debug.log").write_text("\n".join(debug_lines), encoding="utf-8")
        if not all_items:
            raise RuntimeError(
                "查询结果为空：未获取到任何班级记录。请查看 output/bjkbdy_export_debug.log 和 output/bjkbdy_all_items.json"
            )

        print(f"[INFO] 共获取 {len(all_items)} 条，将逐条导出 Excel。")

        kz_url = f"{settings.base_url}/kbdy/bjkbdy_cxXnxqsfkz.html"
        kz_data = _post_json(page, kz_url, {"kzlx": "dy", "xnm": xnm, "xqm": xqm})
        if isinstance(kz_data, str) and "true" in kz_data.lower():
            raise RuntimeError("当前学年学期疑似已控制打印（cxXnxqsfkz 返回 true）")

        xszd = _xszd_map(page, settings, gnmkdm)
        excel_url = f"{settings.base_url}/kbdy/bjkbdy_cxDcExcelBjkb.html?doType=one"

        for idx, item in enumerate(all_items, start=1):
            request_map = _build_pdf_request(item, xszd)
            excel_resp = page.request.post(excel_url, form=request_map)
            if not excel_resp.ok:
                print(f"[WARN] 第 {idx} 条导出失败: HTTP {excel_resp.status}")
                continue
            label = str(item.get("tjkbmc") or item.get("bj") or item.get("bh") or f"item_{idx}")
            path = _save_excel_response(page, settings, output_dir, label, xnm, xqm, excel_resp)
            exported_paths.append(path)

        browser.close()

    return BjBatchResult(total_items=len(all_items), exported_paths=exported_paths)


def run_bjkbdy_export_multi_grades(
    settings: Settings,
    output_dir: Path,
    xnm: str,
    xqm: str,
    grades: list[str],
    gnmkdm: str = "N214505",
    username: str | None = None,
    password: str | None = None,
    headless: bool = False,
) -> BjMultiGradeResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    grade_stats: dict[str, dict[str, int]] = {}

    for grade in grades:
        g = grade.strip()
        if not g:
            continue
        grade_dir = output_dir / g
        grade_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 开始抓取年级 {g} ...")
        res = run_bjkbdy_export_all(
            settings=settings,
            output_dir=grade_dir,
            xnm=xnm,
            xqm=xqm,
            gnmkdm=gnmkdm,
            njdm_id=g,
            username=username,
            password=password,
            headless=headless,
        )
        grade_stats[g] = {"total": res.total_items, "exported": len(res.exported_paths)}
        print(f"[INFO] 年级 {g} 完成：查询 {res.total_items}，导出 {len(res.exported_paths)}")

    return BjMultiGradeResult(grade_stats=grade_stats)


def trace_network(
    settings: Settings,
    output_dir: Path,
    headless: bool = False,
    trace_hint: str = "bjkbdy",
    trace_url: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> list[NetworkRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[NetworkRecord] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(ignore_https_errors=True)
        page = context.new_page()

        def on_response(resp: Any) -> None:
            req = resp.request
            url = req.url
            if trace_hint and trace_hint not in url and "/kbdy/" not in url and "/kbcx/" not in url:
                return

            try:
                text = resp.text()
                text = re.sub(r"\s+", " ", text)[:500]
            except Exception:
                text = "<non-text or unreadable body>"

            content_type = resp.headers.get("content-type", "") if resp.headers else ""
            records.append(
                NetworkRecord(
                    method=req.method,
                    url=url,
                    status=resp.status,
                    content_type=content_type,
                    request_body=req.post_data,
                    response_preview=text,
                )
            )

        context.on("response", on_response)

        def on_new_page(new_page: Page) -> None:
            print(f"[INFO] 检测到新页面: {new_page.url or 'about:blank'}")

        context.on("page", on_new_page)

        _ensure_login(page, settings, username=username, password=password)
        if trace_url:
            page.goto(trace_url, wait_until="domcontentloaded", timeout=settings.timeout_ms)
        else:
            page.goto(f"{settings.base_url}/xtgl/index_initMenu.html", wait_until="domcontentloaded", timeout=settings.timeout_ms)

        print("[INFO] 请在浏览器中点击: 信息查询 -> 班级课表查询，并执行一次查询。")
        print("[INFO] 如果弹出 about:blank，请继续在该窗口触发查询动作。")
        print("[INFO] 操作完成后回到终端按回车，我会再等待 5 秒收尾抓包。")
        input()
        page.wait_for_timeout(5000)

        browser.close()

    trace_path = output_dir / "network_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(
                json.dumps(
                    {
                        "method": r.method,
                        "url": r.url,
                        "status": r.status,
                        "content_type": r.content_type,
                        "request_body": r.request_body,
                        "response_preview": r.response_preview,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    _save_network_summary(records, output_dir)
    return records


def run_crawler(settings: Settings, xnm: str, xqm: str, output_dir: Path, headless: bool = False) -> CrawlResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(ignore_https_errors=True)
        page = context.new_page()

        _ensure_login(page, settings)

        home_url = f"{settings.base_url}/xtgl/index_initMenu.html"
        page.goto(home_url, wait_until="domcontentloaded", timeout=settings.timeout_ms)

        html = page.content()
        csrf = _extract_csrf_token(html)
        raw = _fetch_timetable(page, settings, xnm, xqm, csrf)
        flat_rows = _flatten_records(raw)

        (output_dir / "timetable_raw.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_flat_csv(flat_rows, output_dir / "timetable_flat.csv")

        browser.close()

    return CrawlResult(raw=raw, flat_rows=flat_rows)
