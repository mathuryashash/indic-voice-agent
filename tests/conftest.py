"""Unit tests run against the frozen demo dataset so they never depend on
real VTU dates changing. The official data in app/data is checked by test_official_data.py."""
import os
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent / "eval" / "fixtures" / "demo_data"
os.environ.setdefault("VOICE_AGENT_DATA_DIR", str(DEMO))
