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

## 6. 重建课表 RAG 数据库

当前课表导出文件是旧版二进制 `.xls`。Windows 上可使用 Microsoft ACE OLE DB 将其转换为中间 CSV，再生成结构化 JSON、RAG 语料和 SQLite 数据库：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\export_xls_to_csv.ps1
python .\scripts\build_schedule_corpus.py
python .\scripts\load_schedule_to_sqlite.py
```

默认只处理 `output/2023`、`output/2024`、`output/2025`、`output/2026`：

- `output/_schedule_csv/`：去重后的中间 CSV 和转换清单
- `output/schedule_structured.json`：结构化课程数据
- `output/schedule_rag_corpus.txt`：RAG 文本语料
- `class_schedule.db`：SQLite 数据库

Python 标准库已包含 `sqlite3`，不需要另行安装 SQLite 命令行工具。数据库构建会拒绝包含残缺课程字段的数据。

### 6.1 生成本地向量索引

向量检索使用 FastEmbed 和中文模型 `BAAI/bge-small-zh-v1.5`。首次运行会下载约 90 MB 的模型文件：

```powershell
python -m venv .rag_venv
.\.rag_venv\Scripts\python.exe -m pip install -r .\requirements-rag.txt
.\.rag_venv\Scripts\python.exe .\scripts\build_vector_index.py
```

生成结果位于 `output/vector_store/`：

- `schedule_embeddings.npy`：归一化后的 512 维课表向量
- `schedule_ids.npy`：与 SQLite `courses.id` 对齐的记录 ID
- `manifest.json`：模型、数据条数以及数据库/语料校验哈希

当 `class_schedule.db` 或 `schedule_rag_corpus.txt` 更新后，需要重新运行 `build_vector_index.py`。检索脚本会检查文件哈希，防止使用过期索引。

### 6.2 查询课表向量索引

```powershell
.\.rag_venv\Scripts\python.exe .\scripts\search_schedule.py "230101班星期二第一二节有什么课"
.\.rag_venv\Scripts\python.exe .\scripts\search_schedule.py "王晋茹老师给哪些班上高等数学" --top-k 10
```

查询会自动识别班号、星期、节次和上午/下午，也支持显式过滤：

```powershell
.\.rag_venv\Scripts\python.exe .\scripts\search_schedule.py "高等数学在哪里上课" `
  --grade 2026 --weekday 星期五 --course-name 高等数学 --top-k 5
```

添加 `--json` 可输出适合后端接口消费的 JSON。

## 7. 安全与合规
- 不要把账号密码写入代码或提交到仓库。
- 建议优先通过环境变量或本地终端参数传入凭据。
- 仅抓取你有权限访问的数据，并遵守学校系统使用规范及法律法规。
