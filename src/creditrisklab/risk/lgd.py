"""Loss given default.

LGD = 1 - recovery, blended across the issuer's seniority mix. Every recovery input comes
from a published, cited study (see config/recovery_rates.yaml) and is tagged SOURCED. No
figure in this module is invented, and `config.load_recovery_config` refuses to load a
recovery class that lacks a citation.

Downturn LGD is offered separately from through-the-cycle LGD because they are not the
same number and regulators require the former for capital. Using a long-run average
recovery in a stress scenario understates loss: recoveries fall exactly when default rates
rise (Altman, Brady, Resti & Sironi, 2005, "The Link between Default and Recovery Rates",
Journal of Business 78(6)).
"""

from __future__ import annotations

from typing import Mapping

from creditrisklab.config import load_recovery_config


class LGDError(ValueError):
    pass


def class_lgd(seniority: str, recovery_cfg: dict | None = None, industry_distress: bool = False) -> float:
    cfg = recovery_cfg or load_recovery_config()
    block = cfg["seniority"].get(seniority)
    if block is None:
        raise LGDError(f"unknown seniority class '{seniority}'; add it to recovery_rates.yaml with a citation")
    recovery = float(block["recovery_mean"])

    adj = cfg.get("adjustments", {})
    haircut = adj.get("industry_distress_haircut", {})
    if industry_distress and haircut.get("enabled", False) and seniority in haircut.get("applies_to", []):
        recovery = max(0.0, recovery - float(haircut["value"]))

    lgd = 1.0 - recovery
    floor = float(adj.get("lgd_floor", 0.0))
    cap = float(adj.get("lgd_cap", 1.0))
    return max(floor, min(cap, lgd))


def blended_lgd(
    exposure_mix: Mapping[str, float],
    recovery_cfg: dict | None = None,
    industry_distress: bool = False,
) -> float:
    """Exposure-weighted LGD across the issuer's capital structure."""
    if not exposure_mix:
        raise LGDError("no exposure mix supplied; LGD cannot be blended")
    total = sum(float(v) for v in exposure_mix.values())
    if abs(total - 1.0) > 1e-6:
        raise LGDError(f"exposure weights sum to {total:.4f}, expected 1.0")
    check_single_basis(exposure_mix, recovery_cfg)
    return sum(
        float(weight) * class_lgd(seniority, recovery_cfg, industry_distress)
        for seniority, weight in exposure_mix.items()
    )


def check_single_basis(exposure_mix: Mapping[str, float], recovery_cfg: dict | None = None) -> str:
    """Refuse to blend trading-price and ultimate recoveries into one LGD."""
    cfg = recovery_cfg or load_recovery_config()
    bases = {str(cfg["seniority"].get(s, {}).get("basis", "unspecified")) for s in exposure_mix}
    if len(bases) > 1:
        raise LGDError(
            f"exposure mix combines recovery bases {sorted(bases)}; trading-price and ultimate "
            "recoveries differ materially and cannot be blended into one LGD"
        )
    return bases.pop()


def downturn_lgd(
    exposure_mix: Mapping[str, float],
    recovery_cfg: dict | None = None,
    stress_haircut: float = 0.15,
) -> float:
    """Through-the-cycle LGD with an absolute recovery haircut applied to every class.

    `stress_haircut` is an ASSUMPTION, not a sourced figure, and is labelled as such in the
    report. It is in the range of the recovery decline observed in high-default years in
    Altman et al. (2005) but is not a specific published point estimate.
    """
    cfg = recovery_cfg or load_recovery_config()
    adj = cfg.get("adjustments", {})
    floor, cap = float(adj.get("lgd_floor", 0.0)), float(adj.get("lgd_cap", 1.0))
    total = 0.0
    for seniority, weight in exposure_mix.items():
        block = cfg["seniority"].get(seniority)
        if block is None:
            raise LGDError(f"unknown seniority class '{seniority}'")
        recovery = max(0.0, float(block["recovery_mean"]) - stress_haircut)
        total += float(weight) * max(floor, min(cap, 1.0 - recovery))
    return total


def citations(recovery_cfg: dict | None = None) -> dict[str, str]:
    cfg = recovery_cfg or load_recovery_config()
    return {name: " ".join(str(block["citation"]).split()) for name, block in cfg["seniority"].items()}
