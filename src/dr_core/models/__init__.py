"""The ML core: bounded-error velocity regression, replacing double integration.

OWNER: Sumedha (backup: Sikruti)  |  MILESTONE: M2
Spec: docs/BUILD_PLAN.md section 6.4

The single load-bearing anti-drift decision in this project. Integrating raw
acceleration twice produces metres of error within seconds; a learned model regresses
velocity DIRECTLY, so error stays bounded (the RoNIN / TLIO approach).

Two runtimes, deliberately separated:
  * ``tcn`` -- PyTorch, training only, needs the [ml] extra.
  * ``runtime`` -- ONNX Runtime, inference only, in the default install. The live demo
    machine never needs a 2 GB torch download.
"""

from dr_core.models.runtime import VelocityModelRuntime

__all__ = ["VelocityModelRuntime"]
