"""Serving files that `dr_core.eval.report.generate_report` already wrote.

OWNER: Harsh  |  MILESTONE: M4  |  Spec: docs/BUILD_PLAN.md sections 6.8, 8

No report-*generation* tests here -- that is `tests/test_eval.py`'s job, and Sikruti's
code. This only exercises the read side: given files already on disk, does the route
find them, and does it refuse to be tricked into reading outside `reports_dir`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.gateway import create_app
from services.gateway.reports import content_type_for, resolve_report_file


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "walk_001"
    run_dir.mkdir()
    (run_dir / "report.json").write_text('{"run_id": "walk_001", "drift_pct": 3.2}')
    (run_dir / "trajectory.png").write_bytes(b"not-really-a-png")
    return tmp_path


def test_resolve_report_file_finds_a_real_file(reports_dir: Path) -> None:
    path = resolve_report_file(reports_dir, "walk_001", "report.json")
    assert path is not None
    assert path.read_text() == '{"run_id": "walk_001", "drift_pct": 3.2}'


def test_resolve_report_file_refuses_an_unlisted_filename(reports_dir: Path) -> None:
    """Not on the fixed list `generate_report` actually writes -- refused outright,
    not just a 404, so a typo reads as a bug rather than a missing report."""
    assert resolve_report_file(reports_dir, "walk_001", "../../etc/passwd") is None
    assert resolve_report_file(reports_dir, "walk_001", "notes.txt") is None


def test_resolve_report_file_refuses_a_run_id_that_escapes_reports_dir(
    reports_dir: Path,
) -> None:
    """A run_id like '../../etc' must not let a request read outside reports_dir."""
    assert resolve_report_file(reports_dir, "../../../etc", "report.json") is None


def test_resolve_report_file_is_none_for_a_report_that_does_not_exist_yet(
    reports_dir: Path,
) -> None:
    assert resolve_report_file(reports_dir, "walk_999", "report.json") is None


def test_content_type_is_json_for_the_report_and_png_for_everything_else() -> None:
    assert content_type_for("report.json") == "application/json"
    assert content_type_for("trajectory.png") == "image/png"
    assert content_type_for("nis.png") == "image/png"


def test_reports_route_serves_a_real_file_through_the_app(reports_dir: Path) -> None:
    with TestClient(create_app(reports_dir=reports_dir)) as client:
        response = client.get("/reports/walk_001/report.json")
    assert response.status_code == 200
    assert response.json() == {"run_id": "walk_001", "drift_pct": 3.2}


def test_reports_route_404s_for_a_run_that_was_never_generated(reports_dir: Path) -> None:
    with TestClient(create_app(reports_dir=reports_dir)) as client:
        response = client.get("/reports/never_ran/report.json")
    assert response.status_code == 404


def test_reports_route_is_absent_without_a_reports_dir() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/reports/walk_001/report.json")
    assert response.status_code == 404


def test_healthz_reports_whether_reports_dir_is_configured(reports_dir: Path) -> None:
    with TestClient(create_app(reports_dir=reports_dir)) as client:
        assert client.get("/healthz").json()["reports_dir_configured"] is True
    with TestClient(create_app()) as client:
        assert client.get("/healthz").json()["reports_dir_configured"] is False
