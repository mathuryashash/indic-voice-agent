"""Checks the official VTU data in app/data (loaded in a subprocess so the demo-data
override from conftest does not apply)."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ask(expr: str):
    env = {k: v for k, v in os.environ.items() if k != "VOICE_AGENT_DATA_DIR"}
    code = f"import json; from app import tools as t; print(json.dumps({expr}))"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_bda_credits_and_no_external_lab():
    c = ask("t.get_course_info('big data analytics')")["course"]
    assert c["code"] == "BIS701" and c["credits"] == 4 and c["type"] == "IPCC"
    assert "no separate external lab" in c["lab_assessment"]


def test_course_by_code_and_semester_listing():
    assert ask("t.get_course_info('BIS703')")["course"]["credits"] == 4
    sem = ask("t.get_course_info('', '7')")
    assert sem["total_credits"] == 24 and len(sem["courses"]) == 6
    assert ask("t.get_course_info('machine learning', '6th sem')")["course"]["code"] == "BIS602"


def test_unknown_course():
    assert "error" in ask("t.get_course_info('quantum basket weaving')")


def test_official_exam_dates_carry_warning():
    r = ask("t.get_exam_schedule('BDA')")
    assert r["exam"]["code"] == "BIS701" and r["exam"]["date"] == "2026-11-24"
    assert "likely to change" in r["status_warning"]


def test_calendar_vii_and_vi_not_published():
    vii = ask("t.get_academic_calendar('7', 'exams')")["calendars"][0]
    assert vii["events"]["theory_exams"].startswith("2026-12-02")
    vi = ask("t.get_academic_calendar('6')")["calendars"][0]
    assert vi["year"] == "2025-26" and "not published" in vi["notes"]


def test_holidays_listed():
    r = ask("t.get_academic_calendar('7', 'holidays')")
    assert "IA" in r["note"] and all("date" in h for h in r["upcoming_holidays"])


def test_syllabus_lab_and_textbook():
    hits = ask("t.search_syllabus('Big Data Analytics lab experiments MapReduce')")["results"]
    assert hits[0]["code"] == "BIS701"
    assert ask("t.search_syllabus('CUDA threads blocks grids')")["results"][0]["code"] == "BIS702"
    assert ask("t.search_syllabus('HMAC birthday problem hash')")["results"][0]["code"] == "BIS703"
