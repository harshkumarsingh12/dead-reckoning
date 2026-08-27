"""The frozen contract, and the dependency direction. Both pass from day one.

These are the tests that must NEVER be xfail. If one of them goes red, six people's
assumptions about each other's code have quietly diverged.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import dr_core
from dr_core.types import (
    ERROR_STATE_DIM,
    ERROR_STATE_ORDER,
    FilterState,
    GpsFix,
    HeadingSource,
    ImuSample,
    MagGateVerdict,
    TelemetryFrame,
    VelocityEstimate,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "dr_core"


def test_version_is_exported() -> None:
    assert dr_core.__version__


def test_imu_sample_is_immutable() -> None:
    """Samples are frozen so nothing downstream can edit history in place."""
    s = ImuSample(t_ns=1, a_body=np.zeros(3), w_body=np.zeros(3))
    with pytest.raises(AttributeError):
        s.t_ns = 2  # type: ignore[misc]


def test_timestamps_are_integers() -> None:
    """Nanoseconds as int, never float seconds. Float seconds lose sub-ms precision
    over a long session and that is a silent, unrecoverable class of timing bug."""
    s = ImuSample(t_ns=1_700_000_000_123_456_789, a_body=np.zeros(3), w_body=np.zeros(3))
    assert isinstance(s.t_ns, int)
    assert s.t_ns == 1_700_000_000_123_456_789


def test_error_state_ordering_is_fixed() -> None:
    """Every Jacobian in dr_core.fusion indexes against this. Reordering it silently
    would produce a filter that runs and is wrong."""
    assert ERROR_STATE_ORDER == ("dpx", "dpy", "dvx", "dvy", "dpsi", "db_g", "ds")
    assert ERROR_STATE_DIM == 7


def test_velocity_estimate_reports_sigma() -> None:
    cov = np.diag([0.04, 0.09])
    v = VelocityEstimate(t_ns=0, v_dev=np.array([1.4, 0.0]), cov=cov)
    assert v.sigma_max == pytest.approx(0.3)


def test_filter_state_covariance_is_error_state_sized() -> None:
    state = FilterState(
        t_ns=0,
        p_world=np.zeros(2),
        v_world=np.zeros(2),
        psi_rad=0.0,
        gyro_bias_z=0.0,
        scale=1.0,
        cov=np.eye(ERROR_STATE_DIM),
    )
    assert state.cov.shape == (ERROR_STATE_DIM, ERROR_STATE_DIM)


def test_telemetry_frame_defaults_are_demo_safe() -> None:
    """A frame built with only a state must still render. The UI is built against
    mocks before the filter exists, and a required field added here breaks Tanmay and
    Akshit without warning."""
    state = FilterState(
        t_ns=0,
        p_world=np.zeros(2),
        v_world=np.zeros(2),
        psi_rad=0.0,
        gyro_bias_z=0.0,
        scale=1.0,
        cov=np.eye(ERROR_STATE_DIM),
    )
    frame = TelemetryFrame(t_ns=0, state=state)
    assert frame.nis == {}
    assert frame.zupt_active is False
    assert frame.mag_verdict is MagGateVerdict.REJECTED_INNOVATION


def test_enums_serialise_as_strings() -> None:
    """They cross a WebSocket to a TypeScript client, so the wire value must be the
    readable string and not an integer nobody can debug from the browser console."""
    assert HeadingSource.MAGNETOMETER.value == "magnetometer"
    assert MagGateVerdict.REJECTED_DIP.value == "rejected_dip"


def test_gps_fix_carries_capture_time_not_arrival_time() -> None:
    """Documented on the type itself; asserted here so the docstring cannot rot."""
    assert "CAPTURE time" in (GpsFix.__doc__ or "")


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_dr_core_never_imports_services_or_apps() -> None:
    """Dependencies point inward only.

    dr_core is imported by the training pipeline, the eval harness, and the live
    gateway. The moment it reaches back out to the gateway, training grows a FastAPI
    dependency and the shared-preprocessing guarantee starts to rot.
    """
    offenders: list[str] = []
    for py in SRC.rglob("*.py"):
        for name in _imports_of(py):
            if name.split(".")[0] in {"services", "apps"}:
                offenders.append(f"{py.relative_to(SRC)} imports {name}")
    assert not offenders, "dr_core must not depend on services/ or apps/: " + "; ".join(offenders)


def test_dr_core_does_not_import_torch() -> None:
    """torch is training-only and must never be a hard import.

    The demo laptop installs the default extra. If any module on the live path grows a
    top-level torch import, the gateway stops starting on the one machine that matters.
    """
    offenders = [
        str(py.relative_to(SRC))
        for py in SRC.rglob("*.py")
        if any(n.split(".")[0] == "torch" for n in _imports_of(py))
    ]
    assert not offenders, f"top-level torch import on the live path: {offenders}"
