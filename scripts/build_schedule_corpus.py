from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterable

WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
PERIOD_MAP = {
    "第一二节": "1-2节",
    "第三四节": "3-4节",
    "第五六节": "5-6节",
    "第七八节": "7-8节",
    "第九十节": "9-10节",
    "第十一十二节": "11-12节",
}


def read_csv_rows(csv_path: Path) -> list[list[str]]:
    """Read CSV rows with encoding fallback."""
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    last_error: Exception | None = None
    for enc in encodings:
        try:
            with csv_path.open("r", encoding=enc, newline="") as f:
                return list(csv.reader(f))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to read {csv_path}: {last_error}")


def extract_metadata(first_row: list[str], source_path: Path | None = None) -> tuple[str, str, str, str]:
    """
    Parse semester, major, class_no from first metadata row.
    Example:
    2025-2026年第2学期 250127课表 专业：机械类
    """
    text = " ".join([str(c).strip() for c in first_row if c is not None and str(c).strip()])
    text = re.sub(r"Unnamed:\s*\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    sem_match = re.search(r"(\d{4}\s*-\s*\d{4}年第?\s*\d+学期)", text)
    semester = sem_match.group(1).replace(" ", "") if sem_match else ""

    major = ""
    major_match = re.search(r"专业[:：]\s*(.+)", text)
    if major_match:
        major_raw = major_match.group(1).strip()
        major_raw = re.sub(r"Unnamed:\s*\d+", " ", major_raw, flags=re.IGNORECASE)
        major = re.split(r"[,，。;；\s]+", major_raw, maxsplit=1)[0].strip()

    class_no = ""
    schedule_label = ""
    class_match = re.search(r"((\d{6,8})(?:【[^】]+】)?课表\d*)", text)
    if class_match:
        schedule_label = class_match.group(1).strip()
        class_no = class_match.group(2).strip()
    elif source_path is not None:
        fallback_match = re.match(r"(\d{6,8})", source_path.stem)
        if fallback_match:
            class_no = fallback_match.group(1)
            schedule_label = class_no + "课表"

    return semester, major, class_no, schedule_label


def normalize_cell_text(cell: str) -> str:
    if cell is None:
        return ""
    text = str(cell).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def split_courses(cell_text: str) -> list[str]:
    """One cell can contain multiple courses separated by newlines."""
    return [x.strip() for x in re.split(r"\r?\n+", cell_text) if x.strip()]


def split_course_fields(course_text: str) -> list[str]:
    """
    Split course item by '/' into 6 fields:
    课程名称, 上课周次, 上课地点, 授课教师, 课程代码, 上课班级
    """
    # Split from the right so a slash in a course name (for example TCP/IP)
    # does not shift weeks/location/teacher/course-code fields.
    parts = [p.strip() for p in course_text.rsplit("/", 5)]
    if len(parts) < 6:
        parts += [""] * (6 - len(parts))
    return parts[:6]


def normalize_period(period: str) -> str:
    text = normalize_cell_text(period)
    return PERIOD_MAP.get(text, text)


def split_multi_value(value: str) -> list[str]:
    """Split multi values by common separators and deduplicate in order."""
    text = normalize_cell_text(value)
    if not text:
        return []
    items = [x.strip() for x in re.split(r"[;,；，、]+", text) if x.strip()]
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def parse_weeks_and_period(weeks_raw: str, row_period: str) -> tuple[str, list[int], str, bool]:
    """
    Return:
    - weeks_clean: cleaned weeks expression (without period override prefix)
    - week_list: expanded integer week list
    - period_resolved: final period (override from weeks if present)
    - period_overridden: whether period was overridden
    """
    text = normalize_cell_text(weeks_raw)
    period_resolved = normalize_period(row_period)
    period_overridden = False

    # Detect period override like "(1-2节)11-13周"
    m = re.search(r"^\(([^)]*节[^)]*)\)", text)
    if m:
        period_resolved = normalize_period(m.group(1).strip())
        period_overridden = True
        text = text[m.end() :].strip()

    weeks_clean = text
    # Keep only week numbers/ranges; tolerate formats like "1-16周", "1-4,6,8-10周"
    parse_base = re.sub(r"[周第]", " ", weeks_clean)
    parse_base = re.sub(r"\s+", "", parse_base)

    week_set: set[int] = set()
    for m2 in re.finditer(r"(\d+)\s*-\s*(\d+)|(\d+)", parse_base):
        if m2.group(1) and m2.group(2):
            start = int(m2.group(1))
            end = int(m2.group(2))
            if start <= end:
                for w in range(start, end + 1):
                    week_set.add(w)
            else:
                for w in range(end, start + 1):
                    week_set.add(w)
        elif m2.group(3):
            week_set.add(int(m2.group(3)))

    week_list = sorted(week_set)
    return weeks_clean, week_list, period_resolved, period_overridden


def rag_target_classes(class_no: str, target_list: list[str]) -> str:
    """
    Reduce class-number noise for embedding text.
    Priority:
    1) If current class_no is in list -> use class_no
    2) If only one class -> use it
    3) Else -> "{first}等{n}个班级"
    """
    if not target_list:
        return ""
    if class_no and class_no in target_list:
        return class_no
    if len(target_list) == 1:
        return target_list[0]
    return f"{target_list[0]}等{len(target_list)}个班级"


def iter_grade_csv_files(base_dir: Path, grades: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for grade in grades:
        grade_dir = base_dir / grade
        if grade_dir.exists():
            files.extend(sorted(grade_dir.glob("*.csv")))
    return files


def parse_schedule_csv(csv_path: Path, grade: str, source_file: str) -> tuple[list[dict], list[str]]:
    rows = read_csv_rows(csv_path)
    if len(rows) < 3:
        return [], []

    semester, major, class_no, schedule_label = extract_metadata(rows[0], source_path=csv_path)

    header_text = "".join(rows[1])
    if "星期一" not in header_text and "星期二" not in header_text:
        return [], []

    structured: list[dict] = []
    corpus: list[str] = []

    for row_idx in range(2, len(rows)):
        row = rows[row_idx]
        daytime = normalize_cell_text(row[0] if len(row) > 0 else "")
        period_row = normalize_cell_text(row[1] if len(row) > 1 else "")
        if not period_row:
            continue

        for col_idx, weekday in enumerate(WEEKDAYS, start=2):
            cell = normalize_cell_text(row[col_idx] if len(row) > col_idx else "")
            if not cell:
                continue

            for item_idx, course_item in enumerate(split_courses(cell), start=1):
                try:
                    course_name, weeks_raw, location, teacher_raw, course_code, target_classes_raw = split_course_fields(
                        course_item
                    )
                except IndexError:
                    course_name, weeks_raw, location, teacher_raw, course_code, target_classes_raw = (
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    )

                weeks_clean, week_list, period_real, period_overridden = parse_weeks_and_period(weeks_raw, period_row)
                teacher_list = split_multi_value(teacher_raw)
                target_classes_list = split_multi_value(target_classes_raw)
                remaining_fields = [weeks_raw, location, teacher_raw, course_code, target_classes_raw]
                if not any(remaining_fields):
                    record_type = "block_placeholder"
                elif not all(remaining_fields):
                    record_type = "malformed"
                else:
                    record_type = "course"

                record = {
                    "semester": semester,
                    "major": major,
                    "class_no": class_no,
                    "schedule_label": schedule_label,
                    "grade": grade,
                    "weekday": weekday,
                    "daytime": daytime,
                    "period_original": period_row,
                    "period": period_real,
                    "period_overridden": period_overridden,
                    "course_name": course_name,
                    "weeks_raw": weeks_raw,
                    "weeks": weeks_clean,
                    "week_list": week_list,
                    "location": location,
                    "teacher": teacher_raw,
                    "teacher_list": teacher_list,
                    "course_code": course_code,
                    "target_classes": target_classes_raw,
                    "target_classes_list": target_classes_list,
                    "record_type": record_type,
                    "source_file": source_file,
                    "source_row": row_idx + 1,
                    "source_col": col_idx + 1,
                    "cell_course_index": item_idx,
                    "raw_cell_item": course_item,
                }
                structured.append(record)

                class_prefix = f"{class_no}班" if class_no else ""
                rag_classes = rag_target_classes(class_no, target_classes_list)
                schedule_context = ""
                if schedule_label and schedule_label != f"{class_no}课表":
                    schedule_context = f"，课表标识为{schedule_label}"
                if record_type == "block_placeholder":
                    sentence = (
                        f"{semester}，{class_prefix}{major}专业的学生，在{weekday}{period_real}有《{course_name}》板块课"
                        f"{schedule_context}，具体周次、地点和教师以教务系统选课结果为准。"
                    )
                else:
                    sentence = (
                        f"{semester}，{class_prefix}{major}专业的学生，在{weekday}{period_real}有《{course_name}》课"
                        f"{schedule_context}，上课周次为{weeks_clean}，地点在{location}，授课教师为{teacher_raw}，"
                        f"课程代码为{course_code}，面向班级为{rag_classes}。"
                    )
                corpus.append(sentence)

    return structured, corpus


def load_source_manifest(manifest_path: Path, project_root: Path) -> dict[Path, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mapping: dict[Path, str] = {}
    for item in manifest.get("files", []):
        if item.get("status") != "written":
            continue
        csv_file = str(item.get("csv_file", "")).strip()
        source_file = str(item.get("source_file", "")).strip()
        if csv_file and source_file:
            mapping[(project_root / csv_file).resolve()] = source_file
    return mapping


def validate_structured_records(records: list[dict]) -> None:
    errors: list[str] = []
    seen_coordinates: set[tuple[str, int, int, int]] = set()

    for index, item in enumerate(records, start=1):
        for field in ("semester", "major", "class_no", "schedule_label", "grade", "weekday", "period", "course_name"):
            if not str(item.get(field, "")).strip():
                errors.append(f"record {index}: empty {field}")

        coordinate = (
            str(item.get("source_file", "")),
            int(item.get("source_row", 0)),
            int(item.get("source_col", 0)),
            int(item.get("cell_course_index", 0)),
        )
        if coordinate in seen_coordinates:
            errors.append(f"record {index}: duplicate source coordinate {coordinate}")
        seen_coordinates.add(coordinate)

        if item.get("record_type") == "course":
            for field in ("weeks", "week_list", "location", "teacher", "course_code", "target_classes"):
                if not item.get(field):
                    errors.append(f"record {index}: course has empty {field}")
        elif item.get("record_type") not in {"block_placeholder"}:
            errors.append(f"record {index}: unsupported record_type {item.get('record_type')!r}")

    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"Structured schedule validation failed with {len(errors)} error(s):\n{preview}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build structured schedule data and RAG corpus from converted CSV files.")
    parser.add_argument("--grades", default="2023,2024,2025,2026", help="Comma-separated grade directories.")
    parser.add_argument("--input-dir", type=Path, help="Converted CSV root; defaults to output/_schedule_csv.")
    parser.add_argument("--manifest", type=Path, help="XLS conversion manifest; defaults to <input-dir>/manifest.json.")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output"
    grades = [item.strip() for item in args.grades.split(",") if item.strip()]
    input_dir = (args.input_dir or (output_dir / "_schedule_csv")).resolve()
    manifest_path = (args.manifest or (input_dir / "manifest.json")).resolve()

    if not manifest_path.exists():
        raise SystemExit(f"Conversion manifest not found: {manifest_path}")
    source_mapping = load_source_manifest(manifest_path, project_root=project_root)

    csv_files = iter_grade_csv_files(input_dir, grades)
    if not csv_files:
        raise SystemExit(f"No CSV files found under {input_dir} for grades {','.join(grades)}.")

    all_structured: list[dict] = []
    all_corpus: list[str] = []
    empty_source_files: list[str] = []

    for csv_path in csv_files:
        grade = csv_path.parent.name
        source_file = source_mapping.get(csv_path.resolve())
        if not source_file:
            raise SystemExit(f"CSV file is missing from conversion manifest: {csv_path}")
        structured, corpus = parse_schedule_csv(csv_path, grade=grade, source_file=source_file)
        if not structured:
            empty_source_files.append(source_file)
        all_structured.extend(structured)
        all_corpus.extend(corpus)

    validate_structured_records(all_structured)

    structured_path = output_dir / "schedule_structured.json"
    corpus_path = output_dir / "schedule_rag_corpus.txt"

    structured_path.write_text(json.dumps(all_structured, ensure_ascii=False, indent=2), encoding="utf-8")
    corpus_path.write_text("\n".join(all_corpus), encoding="utf-8")

    print(f"CSV files scanned: {len(csv_files)}")
    print(f"Structured records: {len(all_structured)}")
    print(f"Malformed records: {sum(item['record_type'] == 'malformed' for item in all_structured)}")
    print(f"Block placeholders: {sum(item['record_type'] == 'block_placeholder' for item in all_structured)}")
    print(f"Source files with no scheduled courses: {len(empty_source_files)}")
    for source_file in empty_source_files:
        print(f"  - {source_file}")
    print(f"JSON output: {structured_path}")
    print(f"TXT output: {corpus_path}")


if __name__ == "__main__":
    main()
