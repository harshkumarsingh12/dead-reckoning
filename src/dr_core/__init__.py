"""dr_core — shared core for GPS-denied pedestrian dead reckoning (SIH26168).

This package is imported by BOTH the offline training pipeline and the live demo
pipeline. That is deliberate and load-bearing: any divergence between how training
data and live data are prepared silently degrades the model, and sharing the code
makes the divergence impossible. See docs/BUILD_PLAN.md section 4.

Layer map (see docs/ARCHITECTURE.md):

    timebase    -> one clock domain, reorder buffer
    preprocess  -> calibration, resampling, gravity alignment
    ahrs        -> orientation + magnetometer triple gate
    models      -> learned velocity (causal TCN + NLL covariance)
    fusion      -> 2D ESKF, ZUPT/ZARU, chi-square gating
    baselines   -> raw double integration, PDR
    eval        -> ATE / RTE / drift, NIS / NEES, calibration coverage
    datasets    -> RoNIN / OxIOD / own-recording loaders
    io          -> session record schema

`dr_core` must never import from `services/` or `apps/`. Dependencies point inward
only; tests/test_contract.py enforces this.
"""

from importlib.metadata import version as _installed_version

from dr_core.types import (
    ERROR_STATE_DIM,
    ERROR_STATE_ORDER,
    CarryPosition,
    FilterState,
    GpsFix,
    HeadingSource,
    ImuSample,
    MagGateVerdict,
    OrientationEstimate,
    SessionMeta,
    TelemetryFrame,
    Trajectory,
    VelocityEstimate,
)

# Read from the installed package metadata rather than hardcoded here AND in
# pyproject.toml -- two literals is how a release tag ends up not matching what the
# package actually reports (see .github/workflows/release.yml's version-check step).
# Caveat inherent to editable installs: this reflects pyproject.toml as of the last
# `pip install -e .`, not the working tree this instant -- reinstall after bumping it.
__version__ = _installed_version("dr-core")

__all__ = [
    "ERROR_STATE_DIM",
    "ERROR_STATE_ORDER",
    "CarryPosition",
    "FilterState",
    "GpsFix",
    "HeadingSource",
    "ImuSample",
    "MagGateVerdict",
    "OrientationEstimate",
    "SessionMeta",
    "TelemetryFrame",
    "Trajectory",
    "VelocityEstimate",
    "__version__",
]
