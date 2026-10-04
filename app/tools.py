"""Real, local tools the voice agent can call.

All tools return JSON-serialisable dicts. They never raise on bad user input;
instead they return {"error": "..."} so the LLM can recover and tell the user.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DB = Path(__file__).resolve().parent.parent / "reminders.sqlite"

# ---------------------------------------------------------------- exam schedule

def _load_exams() -> list[dict]:
    return json.loads((DATA_DIR / "exams.json").read_text(encoding="utf-8"))["exams"]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def get_exam_schedule(subject: str = "") -> dict:
    """Return SEE exam date/time/hall for a subject (name, alias or course code).

    Empty subject or 'all' returns the full timetable.
    """
    exams = _load_exams()
    q = _norm(subject or "")
    if q in ("", "all", "all subjects", "every subject", "timetable"):
        return {"exams": exams}
    best, best_score = None, 0.0
    for e in exams:
        names = [e["code"].lower(), _norm(e["subject"]), *[_norm(a) for a in e["aliases"]]]
        for n in names:
            if q == n:
                score = 1.0
            elif re.search(rf"\b{re.escape(n)}\b", q) or re.search(rf"\b{re.escape(q)}\b", n):
                score = 0.9 if len(n) > 3 else 0.75
            else:
                score = SequenceMatcher(None, q, n).ratio()
            if score > best_score:
                best, best_score = e, score
    if best is None or best_score < 0.6:
        return {"error": f"No exam found for '{subject}'.",
                "known_subjects": [e["subject"] for e in exams]}
    return {"exam": best}

# ---------------------------------------------------------------- syllabus search

@dataclass
class Chunk:
    code: str
    subject: str
    module: str
    text: str


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = set("the and of a an to in for on with is are what which how by from at as or".split())


def _tokens(s: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(s.lower()) if t not in _STOP]


class SyllabusIndex:
    def __init__(self, directory: Path = DATA_DIR / "syllabus"):
        self.chunks: list[Chunk] = []
        for path in sorted(directory.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            title = raw.splitlines()[0].lstrip("# ").split(" (")[0]
            code, subject = title.split(" ", 1)
            for part in re.split(r"^## ", raw, flags=re.M)[1:]:
                head, _, body = part.partition("\n")
                self.chunks.append(Chunk(code, subject, head.strip(), " ".join(body.split())))
        # subject name + module title are indexed too so "module 3 of TOC" works
        self._bm25 = BM25Okapi([_tokens(f"{c.code} {c.subject} {c.module} {c.text}") for c in self.chunks])

    def search(self, query: str, k: int = 3) -> list[dict]:
        toks = _tokens(query)
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{"code": self.chunks[i].code, "subject": self.chunks[i].subject,
                 "module": self.chunks[i].module, "text": self.chunks[i].text,
                 "score": round(float(scores[i]), 3)} for i in ranked if scores[i] > 0]


_INDEX: SyllabusIndex | None = None


def search_syllabus(query: str) -> dict:
    """BM25 search over the local syllabus corpus. Query should be in English."""
    global _INDEX
    if _INDEX is None:
        _INDEX = SyllabusIndex()
    hits = _INDEX.search(query or "")
    if not hits:
        return {"error": f"Nothing in the syllabus matched '{query}'."}
    return {"results": hits}

# ---------------------------------------------------------------- reminders

def _connect(db_path: Path | str) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.execute("""CREATE TABLE IF NOT EXISTS reminders (
        id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL,
        due_at TEXT NOT NULL, created_at TEXT NOT NULL)""")
    return con


def set_reminder(text: str, time: str, db_path: Path | str = DEFAULT_DB) -> dict:
    """Store a reminder. `time` must be ISO 8601 local time, e.g. 2026-10-05 18:00."""
    text = (text or "").strip()
    if not text:
        return {"error": "Reminder text is empty."}
    try:
        due = datetime.fromisoformat((time or "").strip().replace("T", " "))
    except ValueError:
        return {"error": f"Could not parse time '{time}'. Use ISO format YYYY-MM-DD HH:MM."}
    with _connect(db_path) as con:
        cur = con.execute("INSERT INTO reminders(text, due_at, created_at) VALUES (?,?,?)",
                          (text, due.strftime("%Y-%m-%d %H:%M"), datetime.now().isoformat(timespec="seconds")))
        rid = cur.lastrowid
    return {"ok": True, "id": rid, "text": text, "due_at": due.strftime("%Y-%m-%d %H:%M")}


def list_reminders(db_path: Path | str = DEFAULT_DB) -> list[dict]:
    with _connect(db_path) as con:
        rows = con.execute("SELECT id, text, due_at FROM reminders ORDER BY due_at").fetchall()
    return [{"id": r[0], "text": r[1], "due_at": r[2]} for r in rows]

# ---------------------------------------------------------------- VTU SGPA / CGPA

# VTU 2022-scheme absolute grading: marks (out of 100) -> grade point
_VTU_GRADES = [(90, 10, "O"), (80, 9, "A+"), (70, 8, "A"), (60, 7, "B+"),
               (55, 6, "B"), (50, 5, "C"), (40, 4, "P"), (0, 0, "F")]


def marks_to_grade_point(marks: float) -> tuple[int, str]:
    if not 0 <= marks <= 100:
        raise ValueError("marks must be between 0 and 100")
    for cutoff, gp, letter in _VTU_GRADES:
        if marks >= cutoff:
            return gp, letter
    raise AssertionError("unreachable")


def calculate_sgpa(courses: list[dict]) -> dict:
    """SGPA = sum(credits_i * grade_point_i) / sum(credits_i).

    Each course: {"credits": number, and either "marks" (0-100) or "grade_point" (0-10)}.
    """
    if not courses:
        return {"error": "No courses given."}
    total_c, total_cg, detail = 0.0, 0.0, []
    try:
        for c in courses:
            credits = float(c["credits"])
            if credits <= 0:
                return {"error": "Credits must be positive."}
            if c.get("grade_point") is not None:
                gp = float(c["grade_point"])
                if not 0 <= gp <= 10:
                    return {"error": "Grade point must be between 0 and 10."}
                letter = None
            elif c.get("marks") is not None:
                gp, letter = marks_to_grade_point(float(c["marks"]))
            else:
                return {"error": "Each course needs marks or grade_point."}
            total_c += credits
            total_cg += credits * gp
            detail.append({"credits": credits, "grade_point": gp, "grade": letter})
    except (KeyError, TypeError, ValueError) as exc:
        return {"error": f"Invalid course data: {exc}"}
    return {"sgpa": round(total_cg / total_c, 2), "total_credits": total_c, "courses": detail}


def calculate_cgpa(semesters: list[dict]) -> dict:
    """CGPA = sum(SGPA_j * credits_j) / sum(credits_j) over semesters.

    Each semester: {"sgpa": number, "credits": number}.
    """
    if not semesters:
        return {"error": "No semesters given."}
    try:
        tc = sum(float(s["credits"]) for s in semesters)
        if tc <= 0:
            return {"error": "Credits must be positive."}
        if any(not 0 <= float(s["sgpa"]) <= 10 for s in semesters):
            return {"error": "SGPA must be between 0 and 10."}
        cg = sum(float(s["sgpa"]) * float(s["credits"]) for s in semesters) / tc
    except (KeyError, TypeError, ValueError) as exc:
        return {"error": f"Invalid semester data: {exc}"}
    return {"cgpa": round(cg, 2), "total_credits": tc,
            "percentage_equivalent": round((cg - 0.75) * 10, 2)}

# ---------------------------------------------------------------- registry + schemas

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "get_exam_schedule",
        "description": "Get the semester-end exam (SEE) date, time and exam hall for a subject. "
                       "Use for any question about when/where an exam is.",
        "parameters": {"type": "object", "properties": {
            "subject": {"type": "string", "description": "Subject name or course code IN ENGLISH, "
                        "e.g. 'Computer Networks', 'TOC', 'BCS501'. Use 'all' for the full timetable."}},
            "required": ["subject"]}}},
    {"type": "function", "function": {
        "name": "search_syllabus",
        "description": "Search the VTU ISE syllabus (Software Engineering, Computer Networks, Theory of "
                       "Computation) for topics, modules, textbooks and credits.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search keywords IN ENGLISH, e.g. 'TCP congestion control module'"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "set_reminder",
        "description": "Save a study reminder for the student at a specific date and time.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "What to remind about, short, in English"},
            "time": {"type": "string", "description": "Local date-time in ISO format YYYY-MM-DD HH:MM"}},
            "required": ["text", "time"]}}},
    {"type": "function", "function": {
        "name": "calculate_sgpa",
        "description": "Compute VTU SGPA for one semester from per-course credits and marks (or grade points).",
        "parameters": {"type": "object", "properties": {
            "courses": {"type": "array", "items": {"type": "object", "properties": {
                "credits": {"type": "number"},
                "marks": {"type": "number", "description": "Total marks out of 100"},
                "grade_point": {"type": "number", "description": "Grade point 0-10, if known instead of marks"}},
                "required": ["credits"]}}},
            "required": ["courses"]}}},
    {"type": "function", "function": {
        "name": "calculate_cgpa",
        "description": "Compute VTU CGPA from each semester's SGPA and credits.",
        "parameters": {"type": "object", "properties": {
            "semesters": {"type": "array", "items": {"type": "object", "properties": {
                "sgpa": {"type": "number"}, "credits": {"type": "number"}},
                "required": ["sgpa", "credits"]}}},
            "required": ["semesters"]}}},
]

TOOLS = {
    "get_exam_schedule": get_exam_schedule,
    "search_syllabus": search_syllabus,
    "set_reminder": set_reminder,
    "calculate_sgpa": calculate_sgpa,
    "calculate_cgpa": calculate_cgpa,
}


def run_tool(name: str, arguments: str | dict, db_path: Path | str | None = None) -> dict:
    """Dispatch a tool call coming from the LLM (arguments may be a JSON string)."""
    if name not in TOOLS:
        return {"error": f"Unknown tool {name}"}
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
    except json.JSONDecodeError:
        return {"error": "Tool arguments were not valid JSON."}
    if name == "set_reminder" and db_path is not None:
        args["db_path"] = db_path
    try:
        return TOOLS[name](**args)
    except TypeError as exc:
        return {"error": f"Bad arguments for {name}: {exc}"}
