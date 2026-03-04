# 教务系统课表抓取（BJUT jwglxt）

用于自动抓取 `https://jwglxt.bjut.edu.cn/` 班级课表，支持：
- 抓包分析接口
- 按条件批量导出班级课表 Excel
- 按多个年级分目录导出

## 1. 环境要求
- Python 3.10+
- Chromium（由 Playwright 安装）

## 2. 安装
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python -m playwright install chromium
```

## 3. 常用命令

### 3.1 抓包分析“班级课表查询”接口
```bash
python -m jiaowu_crawler --trace-network --trace-url "https://jwglxt.bjut.edu.cn/kbdy/bjkbdy_cxBjkbdyIndex.html?gnmkdm=N214505&layout=default" --output ./output
```

### 3.2 导出查询结果全部班级（每页 150 条）
```bash
python -m jiaowu_crawler \
  --bj-export-all \
  --xnm 2025 \
  --xqm 12 \
  --username YOUR_USERNAME \
  --password "YOUR_PASSWORD" \
  --output ./output
```

### 3.3 按多个年级分目录导出（推荐）
```bash
python -m jiaowu_crawler \
  --bj-export-grades 2022,2023,2024,2025 \
  --xnm 2025 \
  --xqm 12 \
  --username YOUR_USERNAME \
  --password "YOUR_PASSWORD" \
  --output ./output
```

输出结构示例：
- `output/2022/*.xls`
- `output/2023/*.xls`
- `output/2024/*.xls`
- `output/2025/*.xls`

## 4. 关键参数
- `--xnm`：学年代码（如 `2025`）
- `--xqm`：学期代码（如 `12`）
- `--gnmkdm`：功能码（默认 `N214505`）
- `--username/--password`：登录凭据

## 5. 调试文件
- `output/bjkbdy_export_debug.log`：分页抓取调试日志
- `output/bjkbdy_all_items.json`：查询到的全部条目
- `output/network_trace.jsonl`：抓包明细
- `output/network_trace_summary.md`：抓包汇总

## 6. 安全与合规
- 不要把账号密码写入代码或提交到仓库。
- 建议优先通过环境变量或本地终端参数传入凭据。
- 仅抓取你有权限访问的数据，并遵守学校系统使用规范及法律法规。
