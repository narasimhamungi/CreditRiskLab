"""CreditRiskLab — PD / LGD / EAD / expected loss on real SEC filing data.

Project 5 of an applied finance/analytics portfolio. Consumes Trellis as an installed
package and does not modify it.
"""

__version__ = "0.1.0"

from creditrisklab.config import load_model_config, load_recovery_config, load_universe  # noqa: E402,F401

__all__ = ["load_model_config", "load_recovery_config", "load_universe", "__version__"]
