from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def build_db(db_path: Path, data_path: Path) -> int:
    """
    Build SQLite DB from schedule_structured.json.
    Returns inserted row count.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"JSON file not found: {data_path}")

    records = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("schedule_structured.json must be a JSON list")
    malformed_count = sum(item.get("record_type") == "malformed" for item in records if isinstance(item, dict))
    if malformed_count:
        raise ValueError(f"Refusing to load {malformed_count} malformed schedule records")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()

        # Idempotent rebuild.
        cur.execute("DROP TABLE IF EXISTS courses")
        cur.execute(
            """
            CREATE TABLE courses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                semester TEXT,
                major TEXT,
                class_no TEXT,
                schedule_label TEXT,
                grade TEXT,
                weekday TEXT,
                daytime TEXT,
                period_original TEXT,
                actual_period TEXT,
                period_overridden INTEGER,
                course_name TEXT,
                weeks_raw TEXT,
                weeks TEXT,
                week_list_json TEXT,
                location TEXT,
                teacher TEXT,
                teacher_list_json TEXT,
                course_code TEXT,
                target_classes TEXT,
                target_class_list_json TEXT,
                record_type TEXT,
                source_file TEXT,
                source_row INTEGER,
                source_col INTEGER,
                cell_course_index INTEGER,
                raw_cell_item TEXT
            )
            """
        )

        rows: list[tuple] = []
        for item in records:
            if not isinstance(item, dict):
                continue

            week_list = item.get("week_list", [])
            teacher_list = item.get("teacher_list", [])
            target_class_list = item.get("target_classes_list", [])

            rows.append(
                (
                    str(item.get("semester", "")),
                    str(item.get("major", "")),
                    str(item.get("class_no", "")),
                    str(item.get("schedule_label", "")),
                    str(item.get("grade", "")),
                    str(item.get("weekday", "")),
                    str(item.get("daytime", "")),
                    str(item.get("period_original", "")),
                    str(item.get("period", "")),
                    1 if bool(item.get("period_overridden", False)) else 0,
                    str(item.get("course_name", "")),
                    str(item.get("weeks_raw", "")),
                    str(item.get("weeks", "")),
                    json.dumps(week_list, ensure_ascii=False),
                    str(item.get("location", "")),
                    str(item.get("teacher", "")),
                    json.dumps(teacher_list, ensure_ascii=False),
                    str(item.get("course_code", "")),
                    str(item.get("target_classes", "")),
                    json.dumps(target_class_list, ensure_ascii=False),
                    str(item.get("record_type", "course")),
                    str(item.get("source_file", "")),
                    int(item.get("source_row", 0) or 0),
                    int(item.get("source_col", 0) or 0),
                    int(item.get("cell_course_index", 0) or 0),
                    str(item.get("raw_cell_item", "")),
                )
            )

        cur.executemany(
            """
            INSERT INTO courses (
                semester, major, class_no, schedule_label, grade, weekday, daytime,
                period_original, actual_period, period_overridden,
                course_name, weeks_raw, weeks, week_list_json,
                location, teacher, teacher_list_json, course_code,
                target_classes, target_class_list_json,
                record_type, source_file, source_row, source_col, cell_course_index, raw_cell_item
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        # Indexes for high-frequency queries.
        cur.execute("CREATE INDEX idx_courses_course_name ON courses(course_name)")
        cur.execute("CREATE INDEX idx_courses_weekday ON courses(weekday)")
        cur.execute("CREATE INDEX idx_courses_major ON courses(major)")
        cur.execute("CREATE INDEX idx_courses_class_no ON courses(class_no)")
        cur.execute("CREATE INDEX idx_courses_teacher ON courses(teacher)")
        cur.execute("CREATE INDEX idx_courses_grade ON courses(grade)")
        cur.execute("CREATE INDEX idx_courses_semester ON courses(semester)")
        cur.execute("CREATE INDEX idx_courses_actual_period ON courses(actual_period)")
        cur.execute("CREATE INDEX idx_courses_course_code ON courses(course_code)")
        cur.execute("CREATE INDEX idx_courses_record_type ON courses(record_type)")
        cur.execute(
            "CREATE UNIQUE INDEX idx_courses_source_coordinate "
            "ON courses(source_file, source_row, source_col, cell_course_index)"
        )

        conn.commit()
        return len(rows)
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load structured schedule JSON into SQLite.")
    parser.add_argument("--db-path", type=Path, help="SQLite output path; defaults to class_schedule.db.")
    parser.add_argument("--data-path", type=Path, help="Structured JSON path; defaults to output/schedule_structured.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    db_path = (args.db_path or (project_root / "class_schedule.db")).resolve()
    data_path = (args.data_path or (project_root / "output" / "schedule_structured.json")).resolve()

    inserted = build_db(db_path=db_path, data_path=data_path)
    print(f"Inserted rows: {inserted}")
    print(f"SQLite DB: {db_path}")


if __name__ == "__main__":
    main()
