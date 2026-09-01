# LLM Agent for University-Wide Course Schedule Search

A timetable crawler and local retrieval pipeline for Beijing University of Technology (BJUT). The project collects class schedules from the university's `jwglxt` academic administration system and turns the exported data into a searchable SQLite database and local vector index.

## Features

- Inspect network requests used by the timetable search page.
- Export all matching class schedules to Excel in configurable batches.
- Organize exported schedules into separate directories by admission year.
- Convert legacy `.xls` files into structured schedule records.
- Build a local SQLite database and RAG-ready text corpus.
- Create and query a Chinese-language vector index for natural-language schedule search.

## Requirements

- Python 3.10 or later
- Chromium, installed through Playwright
- Microsoft ACE OLE DB Provider on Windows when converting legacy `.xls` files

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m playwright install chromium
```

## Timetable Crawling

### Inspect timetable network requests

```powershell
python -m jiaowu_crawler `
  --trace-network `
  --trace-url "https://jwglxt.bjut.edu.cn/kbdy/bjkbdy_cxBjkbdyIndex.html?gnmkdm=N214505&layout=default" `
  --output .\output
```

### Export every class in the current query

The crawler requests up to 150 records per page and exports every matching class:

```powershell
python -m jiaowu_crawler `
  --bj-export-all `
  --xnm 2025 `
  --xqm 12 `
  --username YOUR_USERNAME `
  --password "YOUR_PASSWORD" `
  --output .\output
```

### Export schedules by admission year

```powershell
python -m jiaowu_crawler `
  --bj-export-grades 2022,2023,2024,2025 `
  --xnm 2025 `
  --xqm 12 `
  --username YOUR_USERNAME `
  --password "YOUR_PASSWORD" `
  --output .\output
```

Example output:

```text
output/
├── 2022/
├── 2023/
├── 2024/
└── 2025/
```

### Important arguments

- `--xnm`: academic year code, such as `2025`
- `--xqm`: semester code, such as `12`
- `--gnmkdm`: function code; defaults to `N214505`
- `--username` and `--password`: university system credentials

### Debugging artifacts

The crawler may produce the following files under `output/`:

- `bjkbdy_export_debug.log`: pagination and export diagnostics
- `bjkbdy_all_items.json`: all records returned by the timetable query
- `network_trace.jsonl`: captured network request details
- `network_trace_summary.md`: a readable network trace summary

## Local Schedule Retrieval Pipeline

The exported schedules use the legacy binary `.xls` format. On Windows, convert them to intermediate CSV files before generating the structured dataset, RAG corpus, and SQLite database:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\export_xls_to_csv.ps1
python .\scripts\build_schedule_corpus.py
python .\scripts\load_schedule_to_sqlite.py
```

By default, the scripts process `output/2023`, `output/2024`, `output/2025`, and `output/2026`.

Generated artifacts:

- `output/_schedule_csv/`: deduplicated intermediate CSV files and conversion manifest
- `output/schedule_structured.json`: structured schedule records
- `output/schedule_rag_corpus.txt`: RAG text corpus
- `class_schedule.db`: SQLite schedule database

Python includes the `sqlite3` module, so the SQLite command-line application is not required. The database builder rejects records with incomplete course fields.

## Vector Search

Vector retrieval uses FastEmbed and the Chinese embedding model `BAAI/bge-small-zh-v1.5`. The first run downloads approximately 90 MB of model files.

### Build the vector index

```powershell
python -m venv .rag_venv
.\.rag_venv\Scripts\python.exe -m pip install -r .\requirements-rag.txt
.\.rag_venv\Scripts\python.exe .\scripts\build_vector_index.py
```

The vector store is written to `output/vector_store/`:

- `schedule_embeddings.npy`: normalized 512-dimensional schedule embeddings
- `schedule_ids.npy`: record IDs aligned with `courses.id` in SQLite
- `manifest.json`: model metadata, record counts, and source-file hashes

Rebuild the index whenever `class_schedule.db` or `schedule_rag_corpus.txt` changes. The search command validates file hashes to prevent queries against a stale index.

### Search the schedule index

```powershell
.\.rag_venv\Scripts\python.exe .\scripts\search_schedule.py "230101班星期二第一二节有什么课"
.\.rag_venv\Scripts\python.exe .\scripts\search_schedule.py "王晋茹老师给哪些班上高等数学" --top-k 10
```

The query parser recognizes class numbers, weekdays, periods, and time-of-day expressions. Filters can also be provided explicitly:

```powershell
.\.rag_venv\Scripts\python.exe .\scripts\search_schedule.py "高等数学在哪里上课" `
  --grade 2026 `
  --weekday 星期五 `
  --course-name 高等数学 `
  --top-k 5
```

Add `--json` to produce machine-readable output suitable for an API or agent backend.

## Dual-Lane RAG API

The API keeps the deterministic local search path and adds a DeepSeek-assisted path:

1. The direct lane parses obvious class, weekday, and period filters and searches the local vectors immediately.
2. In parallel, DeepSeek rewrites the question and extracts additional structured filters; the rewritten query searches the same local index.
3. The backend fuses both rankings with reciprocal rank fusion, then asks DeepSeek to answer only from the fused evidence.
4. If the API key is absent, DeepSeek times out, or model output is invalid, the endpoint still returns local results and a deterministic summary.

Install and configure the backend:

```powershell
.\.rag_venv\Scripts\python.exe -m pip install -r .\requirements-backend.txt
Copy-Item .env.example .env
# Edit .env and set DEEPSEEK_API_KEY. Never commit the real key.
```

Start it from the project root:

```powershell
$env:PYTHONPATH = "src"
.\.rag_venv\Scripts\python.exe -m uvicorn jiaowu_rag.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API page, or query it directly:

```powershell
$body = @{
  question = "230101班星期二第一二节有什么课"
  top_k = 5
  use_deepseek = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/query `
  -ContentType application/json `
  -Body $body
```

`GET /health` reports the vector record count and whether DeepSeek is configured. Set `use_deepseek` to `false` on an individual request to force local-only retrieval.

## Security and Responsible Use

- Never store university usernames or passwords in source code or commit them to Git.
- Prefer environment variables or local command-line arguments for credentials.
- Access only data you are authorized to use.
- Follow university policies and all applicable laws and regulations.
