import pytest

from app import tools


def test_exam_by_alias_and_code():
    assert tools.get_exam_schedule("CN")["exam"]["code"] == "BCS502"
    assert tools.get_exam_schedule("computer networks exam")["exam"]["date"] == "2026-12-14"
    assert tools.get_exam_schedule("bcs503")["exam"]["subject"] == "Theory of Computation"
    assert tools.get_exam_schedule("Software Engineering")["exam"]["code"] == "BCS501"


def test_exam_fuzzy_and_unknown():
    assert tools.get_exam_schedule("theory of computaton")["exam"]["code"] == "BCS503"
    out = tools.get_exam_schedule("quantum basket weaving")
    assert "error" in out and "Computer Networks" in out["known_subjects"]


def test_exam_all():
    assert len(tools.get_exam_schedule("all")["exams"]) == 4


@pytest.mark.parametrize("q,code,module_word", [
    ("TCP congestion control", "BCS502", "Transport"),
    ("pumping lemma regular languages", "BCS503", "Regular"),
    ("COCOMO estimation", "BCS501", "Estimation"),
    ("Turing machine halting problem", "BCS503", "Turing"),
    ("scrum sprint", "BCS501", "Agile"),
    ("distance vector routing", "BCS502", "Network"),
])
def test_syllabus_bm25_top1(q, code, module_word):
    top = tools.search_syllabus(q)["results"][0]
    assert top["code"] == code and module_word in top["module"]


def test_syllabus_no_match():
    assert "error" in tools.search_syllabus("zzzz qqqq")


def test_reminder_roundtrip(tmp_path):
    db = tmp_path / "r.sqlite"
    out = tools.set_reminder("revise TOC module 3", "2026-10-05 18:00", db_path=db)
    assert out["ok"] and out["due_at"] == "2026-10-05 18:00"
    assert tools.list_reminders(db) == [{"id": 1, "text": "revise TOC module 3", "due_at": "2026-10-05 18:00"}]


def test_reminder_bad_input(tmp_path):
    db = tmp_path / "r.sqlite"
    assert "error" in tools.set_reminder("x", "tomorrow evening", db_path=db)
    assert "error" in tools.set_reminder("", "2026-10-05 18:00", db_path=db)
    assert tools.list_reminders(db) == []


@pytest.mark.parametrize("marks,gp,letter", [(100, 10, "O"), (90, 10, "O"), (89, 9, "A+"), (75, 8, "A"),
                                             (60, 7, "B+"), (57, 6, "B"), (50, 5, "C"), (40, 4, "P"), (39, 0, "F")])
def test_vtu_grade_points(marks, gp, letter):
    assert tools.marks_to_grade_point(marks) == (gp, letter)


def test_sgpa_formula():
    # (4*10 + 4*9 + 3*8) / 11 = 100/11 = 9.09
    out = tools.calculate_sgpa([{"credits": 4, "marks": 92}, {"credits": 4, "marks": 85}, {"credits": 3, "marks": 72}])
    assert out["sgpa"] == 9.09 and out["total_credits"] == 11
    assert tools.calculate_sgpa([{"credits": 3, "grade_point": 8}, {"credits": 1, "grade_point": 10}])["sgpa"] == 8.5


def test_sgpa_errors():
    assert "error" in tools.calculate_sgpa([])
    assert "error" in tools.calculate_sgpa([{"credits": 4}])
    assert "error" in tools.calculate_sgpa([{"credits": 4, "marks": 120}])
    assert "error" in tools.calculate_sgpa([{"credits": 0, "marks": 50}])


def test_cgpa_formula():
    out = tools.calculate_cgpa([{"sgpa": 8.0, "credits": 20}, {"sgpa": 9.0, "credits": 20}])
    assert out["cgpa"] == 8.5 and out["percentage_equivalent"] == 77.5
    assert "error" in tools.calculate_cgpa([{"sgpa": 11, "credits": 20}])


def test_run_tool_dispatch(tmp_path):
    assert tools.run_tool("get_exam_schedule", '{"subject": "TOC"}')["exam"]["code"] == "BCS503"
    assert "error" in tools.run_tool("nope", "{}")
    assert "error" in tools.run_tool("get_exam_schedule", "not json")
    assert "error" in tools.run_tool("get_exam_schedule", '{"wrong": 1}')
    r = tools.run_tool("set_reminder", {"text": "a", "time": "2026-10-05 07:00"}, db_path=tmp_path / "x.sqlite")
    assert r["ok"]


def test_schemas_match_registry():
    assert {s["function"]["name"] for s in tools.TOOL_SCHEMAS} == set(tools.TOOLS)


def test_syllabus_overview_chunk_has_textbooks_and_credits():
    top = tools.search_syllabus("Theory of Computation textbook")["results"][0]
    assert top["code"] == "BCS503" and "Hopcroft" in top["text"]
    assert "Credits: 4" in tools.search_syllabus("software engineering credits")["results"][0]["text"]
