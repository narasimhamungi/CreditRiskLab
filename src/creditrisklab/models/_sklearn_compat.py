"""Version shim for `LogisticRegression`.

scikit-learn 1.8 deprecated the `penalty` argument in favour of `l1_ratio` / `C`, with
removal scheduled for 1.10. Pinning an old scikit-learn to dodge this would leave the repo
running deprecated APIs; passing `penalty` unconditionally floods every run with warnings
and breaks outright on 1.10. The shim translates at call time so the same code runs on both
sides of the change.
"""

from __future__ import annotations

import inspect

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

_PARAMS = inspect.signature(LogisticRegression.__init__).parameters


def _version_tuple() -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in str(sklearn.__version__).split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


# `penalty` still exists in 1.8 but warns on every call, so the new dialect is preferred
# from 1.8 onward rather than only once `penalty` is removed.
_USE_L1_RATIO = "l1_ratio" in _PARAMS and _version_tuple() >= (1, 8)
_SUPPORTS_PENALTY = "penalty" in _PARAMS and not _USE_L1_RATIO
UNPENALISED_C = 1e12


def logistic_regression(penalty: str | None = "l2", C: float = 1.0, **kwargs) -> LogisticRegression:
    """Construct a LogisticRegression with penalty expressed in whichever dialect is available."""
    if _SUPPORTS_PENALTY:
        return LogisticRegression(penalty=penalty, C=C, **kwargs)

    # New dialect: no `penalty`. l1_ratio selects the mix. sklearn's recommended C=np.inf
    # for "unpenalised" emits a UserWarning on every fit in 1.8; C=1e12 with an l2 mix is
    # numerically indistinguishable for the calibration-slope regression it is used for.
    if penalty is None:
        return LogisticRegression(C=UNPENALISED_C, l1_ratio=0.0, **kwargs)
    if penalty == "l1":
        return LogisticRegression(C=C, l1_ratio=1.0, **kwargs)
    if penalty == "elasticnet":
        return LogisticRegression(C=C, l1_ratio=float(kwargs.pop("l1_ratio", 0.5)), **kwargs)
    return LogisticRegression(C=C, l1_ratio=0.0, **kwargs)


def unpenalised_logistic_regression(**kwargs) -> LogisticRegression:
    return logistic_regression(penalty=None, **kwargs)
