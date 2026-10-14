from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

DASHBOARD_FILE = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")


def test_dashboard_initializes_cleanly():
    """Verify that the dashboard loads, sets title, and initializes without uncaught exceptions."""
    at = AppTest.from_file(DASHBOARD_FILE, default_timeout=15)
    at.run()
    assert len(at.exception) == 0
    # Confirm sidebar configuration inputs exist
    assert len(at.sidebar.text_input) >= 1
    assert len(at.sidebar.selectbox) >= 2

