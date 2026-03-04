from __future__ import annotations

import argparse
from pathlib import Path

from .client import run_bjkbdy_export_all, run_bjkbdy_export_multi_grades, run_bjkbdy_pdf, run_crawler, trace_network
from .config import load_settings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BJUT jwglxt timetable crawler")
    p.add_argument("--xnm", required=False, help="学年代码，例如 2025")
    p.add_argument("--xqm", required=False, help="学期代码，例如 1 或 2")
    p.add_argument("--output", default="output", help="输出目录")
    p.add_argument("--base-url", default=None, help="教务系统地址，默认 https://jwglxt.bjut.edu.cn")
    p.add_argument("--headless", action="store_true", help="使用无头模式")
    p.add_argument("--trace-network", action="store_true", help="记录班级课表相关网络请求与响应摘要")
    p.add_argument("--trace-hint", default="bjkbdy", help="网络请求过滤关键词，默认 bjkbdy")
    p.add_argument("--trace-url", default=None, help="登录后直接跳转到指定页面进行抓包")
    p.add_argument("--username", default=None, help="登录用户名（学号/工号）")
    p.add_argument("--password", default=None, help="登录密码")
    p.add_argument("--bj-class-name", default=None, help="按班级名导出班级课表 PDF")
    p.add_argument("--bj-export-all", action="store_true", help="查询后按每页150条逐条导出全部班级Excel")
    p.add_argument("--bj-export-grades", default=None, help="按年级批量导出，逗号分隔，如 2022,2023,2024,2025")
    p.add_argument("--njdm-id", default=None, help="班级所属年级代码，例如 2022")
    p.add_argument("--xqh-id", default=None, help="校区代码，例如 1")
    p.add_argument("--gnmkdm", default="N214505", help="功能码，默认 N214505")
    return p


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings(base_url_override=args.base_url)

    if args.trace_network:
        records = trace_network(
            settings=settings,
            output_dir=Path(args.output),
            headless=args.headless,
            trace_hint=args.trace_hint,
            trace_url=args.trace_url,
            username=args.username,
            password=args.password,
        )
        print(f"[OK] 抓包完成，请求数: {len(records)}")
        print(f"[OK] 结果文件: {(Path(args.output) / 'network_trace.jsonl').resolve()}")
        print(f"[OK] 汇总文件: {(Path(args.output) / 'network_trace_summary.md').resolve()}")
        return

    if args.bj_export_all:
        if not args.xnm or not args.xqm:
            raise SystemExit("使用 --bj-export-all 时，必须提供 --xnm --xqm")
        result = run_bjkbdy_export_all(
            settings=settings,
            output_dir=Path(args.output),
            xnm=str(args.xnm),
            xqm=str(args.xqm),
            gnmkdm=str(args.gnmkdm),
            username=args.username,
            password=args.password,
            headless=args.headless,
        )
        print(f"[OK] 查询条数: {result.total_items}")
        print(f"[OK] 导出成功: {len(result.exported_paths)}")
        print(f"[OK] 输出目录: {Path(args.output).resolve()}")
        return

    if args.bj_export_grades:
        if not args.xnm or not args.xqm:
            raise SystemExit("使用 --bj-export-grades 时，必须提供 --xnm --xqm")
        grades = [x.strip() for x in str(args.bj_export_grades).split(",") if x.strip()]
        result = run_bjkbdy_export_multi_grades(
            settings=settings,
            output_dir=Path(args.output),
            xnm=str(args.xnm),
            xqm=str(args.xqm),
            grades=grades,
            gnmkdm=str(args.gnmkdm),
            username=args.username,
            password=args.password,
            headless=args.headless,
        )
        print("[OK] 多年级导出完成：")
        for g, stat in result.grade_stats.items():
            print(f"[OK] {g}: 查询 {stat['total']}, 导出 {stat['exported']}")
        print(f"[OK] 输出目录: {Path(args.output).resolve()}")
        return

    if args.bj_class_name:
        if not args.xnm or not args.xqm:
            raise SystemExit("使用 --bj-class-name 时，必须提供 --xnm --xqm")
        result = run_bjkbdy_pdf(
            settings=settings,
            output_dir=Path(args.output),
            class_name=args.bj_class_name,
            xnm=str(args.xnm),
            xqm=str(args.xqm),
            njdm_id=str(args.njdm_id),
            xqh_id=str(args.xqh_id),
            gnmkdm=str(args.gnmkdm),
            username=args.username,
            password=args.password,
            headless=args.headless,
        )
        print(f"[OK] 班级: {result.class_name}")
        print(f"[OK] Excel: {result.pdf_path.resolve()}")
        print(f"[OK] 班级详情: {(Path(args.output) / 'bjkbdy_selected_item.json').resolve()}")
        return

    if not args.xnm or not args.xqm:
        raise SystemExit("未使用 --trace-network 时，必须提供 --xnm 和 --xqm")

    result = run_crawler(
        settings=settings,
        xnm=str(args.xnm),
        xqm=str(args.xqm),
        output_dir=Path(args.output),
        headless=args.headless,
    )
    print(f"[OK] 抓取完成，课程记录数: {len(result.flat_rows)}")
    print(f"[OK] 输出目录: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
