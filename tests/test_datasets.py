"""Unit tests for dataset loading: our own recordings and the trajectory split.

Spec: docs/BUILD_PLAN.md section 4  |  OWNER: Sumedha  |  MILESTONE: M0

RoNIN/OxIOD parsing itself is not tested here -- it is not implemented (no access, no
real file to verify a schema against; see loaders.py). These cover what IS implemented:
the access-request guard, our own session format round-trip, the Recording properties,
and the trajectory-level split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dr_core.datasets import (
    Recording,
    load_own_recording,
    load_oxiod,
    load_ronin,
    split_by_trajectory,
)
from dr_core.io import SessionWriter
from dr_core.types import CarryPosition, GpsFix, ImuSample, SessionMeta, Trajectory

NS_PER_S = 1_000_000_000


def _session_meta(**overrides: object) -> SessionMeta:
    defaults: dict[str, object] = {
        "session_id": "test-own-0001",
        "device_model": "synthetic",
        "carry_position": CarryPosition.HAND,
        "imu_rate_hz": 200.0,
        "origin_lat_deg": 20.3535,
        "origin_lon_deg": 85.8164,
    }
    defaults.update(overrides)
    return SessionMeta(**defaults)  # type: ignore[arg-type]


def test_load_ronin_raises_with_the_access_url_when_root_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"ronin\.cs\.sfu\.ca"):
        load_ronin(tmp_path / "does-not-exist")


def test_load_oxiod_raises_with_the_access_url_when_root_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"deepio\.cs\.ox\.ac\.uk"):
        load_oxiod(tmp_path / "does-not-exist")


def test_load_ronin_stops_short_of_a_guess_once_the_root_exists(tmp_path: Path) -> None:
    """Root exists (access granted) but the real on-disk schema is still unverified --
    must fail loudly, not silently return a plausible-looking wrong parse."""
    (tmp_path / "ronin_root").mkdir()
    with pytest.raises(NotImplementedError):
        load_ronin(tmp_path / "ronin_root")


def test_load_own_recording_round_trips_imu_and_gps(tmp_path: Path) -> None:
    meta = _session_meta()
    path = tmp_path / "session.jsonl.gz"
    with SessionWriter(path, meta) as writer:
        for i in range(5):
            writer.write_imu(
                ImuSample(
                    t_ns=i * (NS_PER_S // 200),
                    a_body=np.array([0.0, 0.0, 9.80665]),
                    w_body=np.zeros(3),
                )
            )
        # A short, deliberately non-zero walk east, so the ENU projection is checkable.
        writer.write_gps(
            GpsFix(
                t_ns=0,
                lat_deg=meta.origin_lat_deg or 0.0,
                lon_deg=meta.origin_lon_deg or 0.0,
                accuracy_m=3.0,
            )
        )
        writer.write_gps(
            GpsFix(
                t_ns=1 * NS_PER_S,
                lat_deg=meta.origin_lat_deg or 0.0,
                lon_deg=(meta.origin_lon_deg or 0.0) + 0.0001,
                accuracy_m=3.0,
            )
        )

    recording = load_own_recording(path)

    assert recording.meta.session_id == "test-own-0001"
    assert len(recording.imu) == 5
    assert len(recording.gps) == 2
    assert recording.truth is not None
    assert recording.truth.label == "gps_truth"
    # First fix sits exactly on the origin -> world ENU (0, 0).
    np.testing.assert_allclose(recording.truth.p_world[0], [0.0, 0.0], atol=1e-6)
    # Moving east (increasing longitude) must increase the world x (East) coordinate.
    assert recording.truth.p_world[1, 0] > 0.0
    assert abs(recording.truth.p_world[1, 1]) < 1.0  # negligible north drift


def test_load_own_recording_has_no_truth_without_gps(tmp_path: Path) -> None:
    meta = _session_meta()
    path = tmp_path / "no_gps.jsonl.gz"
    with SessionWriter(path, meta) as writer:
        writer.write_imu(ImuSample(t_ns=0, a_body=np.zeros(3), w_body=np.zeros(3)))

    recording = load_own_recording(path)
    assert recording.truth is None


def _recording(label: str, n_points: int = 10) -> Recording:
    meta = _session_meta(session_id=label)
    imu = [
        ImuSample(t_ns=i * (NS_PER_S // 200), a_body=np.zeros(3), w_body=np.zeros(3))
        for i in range(20)
    ]
    truth = Trajectory(
        t_ns=np.arange(n_points, dtype=np.int64) * NS_PER_S,
        p_world=np.column_stack([np.arange(n_points, dtype=np.float64), np.zeros(n_points)]),
        label=label,
    )
    return Recording(meta=meta, imu=imu, truth=truth)


def test_recording_duration_and_distance() -> None:
    r = _recording("r0")
    assert r.duration_s == pytest.approx(19 / 200)
    assert r.distance_m == pytest.approx(9.0)  # 10 points, 1 m apart


def test_split_by_trajectory_partitions_every_recording_exactly_once() -> None:
    recordings = [_recording(f"r{i}") for i in range(20)]
    train, val, test = split_by_trajectory(recordings, train_frac=0.7, val_frac=0.15, seed=26168)

    assert len(train) + len(val) + len(test) == len(recordings)
    all_ids = {r.meta.session_id for r in train + val + test}
    assert all_ids == {r.meta.session_id for r in recordings}
    # No id appears in more than one split -- a trajectory is never cut across sets.
    assert len(all_ids) == len(train) + len(val) + len(test)


def test_split_by_trajectory_is_deterministic_for_a_fixed_seed() -> None:
    recordings = [_recording(f"r{i}") for i in range(20)]
    split_a = split_by_trajectory(recordings, seed=26168)
    split_b = split_by_trajectory(recordings, seed=26168)
    for a, b in zip(split_a, split_b, strict=True):
        assert [r.meta.session_id for r in a] == [r.meta.session_id for r in b]


def test_split_by_trajectory_rejects_invalid_fractions() -> None:
    with pytest.raises(ValueError, match="invalid split"):
        split_by_trajectory([_recording("r0")], train_frac=0.8, val_frac=0.5)
