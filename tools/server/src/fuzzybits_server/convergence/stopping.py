"""The controller's stopping specification, its validation, and the legacy mapping.

This module mirrors the stopping specification fixed for the controller library
(#224) so the server can accept and validate the same model the controller will
execute in phase 2:

============  ====================  ==================================================
field         #224 flag             meaning
============  ====================  ==================================================
rel_error     ``--rel-error``       goal: reported half-width <= ``rel_error * p_hat``
abs_error     ``--abs-error``       goal: reported half-width <= ``abs_error`` (a floor)
confidence    ``--confidence``      interval level for the goal and the report (0.95)
max_trials    ``--max-trials``      limit: stop after this many committed trials
max_time      ``--max-time``        limit: stop after this duration (``none`` = no limit)
trials        ``-N``                design: exactly this many trials, no early stop
goal_queries  ``--goal-query``      which queries must meet the goal
require_goal  ``--require-goal``    a limit before the goal is a failure
============  ====================  ==================================================

Resolution follows the same rules: a goal run needs no ``rel_error`` (default
``0.01``), ``confidence`` defaults to ``0.95``, and ``max_time`` defaults to ten
minutes in goal mode and to no limit for a fixed design. A specification that
selects nothing at all is a goal run with those defaults; an explicitly
unbounded run (legacy ``num_trials = 0`` with ``early_stop = false``) has to say
``max_time: "none"``.

Phase 1 executes only the subset the host-side loop can honour faithfully:
a fixed design, a trial limit, a time limit, and a goal evaluated on the Wilson
interval with the ``max(eps*p_hat, a)`` rule. Everything else is rejected with
:class:`UnsupportedSpecError` rather than approximated; see
``tools/server/README.md``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, model_validator

DEFAULT_REL_ERROR = 0.01
DEFAULT_CONFIDENCE = 0.95
DEFAULT_GOAL_MAX_TIME_S = 600.0

#: The controller library is the only implementation that can honour these.
CONTROLLER_REQUIRED = "requires the controller library, #224"

_DURATION = re.compile(r"^\s*(?P<value>[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*(?P<unit>ms|s|sec|secs|m|min|mins|h|hr|hrs)?\s*$")
_DURATION_UNITS = {None: 1.0, "ms": 1.0e-3, "s": 1.0, "sec": 1.0, "secs": 1.0, "m": 60.0, "min": 60.0, "mins": 60.0, "h": 3600.0, "hr": 3600.0, "hrs": 3600.0}
_NO_LIMIT = {"none", "off", "no-limit", "nolimit", "unbounded", "infinite", "inf", ""}


class StoppingSpecError(ValueError):
    """A stopping specification that does not validate (#224 rule, no execution)."""


class UnsupportedSpecError(StoppingSpecError):
    """A valid specification the host-side loop cannot honour faithfully."""


def parse_duration(value: Any) -> float | None:
    """Parse ``90s`` / ``10m`` / ``1.5h`` / a number of seconds / ``none``.

    Returns ``None`` for an explicit "no limit" and raises
    :class:`StoppingSpecError` for anything else that is not a positive duration.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise StoppingSpecError("max_time must be a duration like 90s, 10m or 2h, or none")
    if isinstance(value, (int, float)):
        seconds = float(value)
        if not math.isfinite(seconds) or seconds <= 0.0:
            raise StoppingSpecError("max_time must be positive, or none for no time limit")
        return seconds
    text = str(value).strip().lower()
    if text in _NO_LIMIT:
        return None
    match = _DURATION.match(text)
    if match is None:
        if text.startswith("-") and _DURATION.match(text[1:]):
            raise StoppingSpecError("max_time must be positive, or none for no time limit")
        raise StoppingSpecError(
            f"max_time {value!r} is not a duration; write seconds or a value like 90s, 10m, 2h, or none"
        )
    seconds = float(match.group("value")) * _DURATION_UNITS[match.group("unit")]
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise StoppingSpecError("max_time must be positive, or none for no time limit")
    return seconds


def format_duration(seconds: float | None) -> str:
    """Render a duration the way the #224 flags accept it."""
    if seconds is None:
        return "none"
    for unit, size in (("h", 3600.0), ("m", 60.0)):
        if seconds >= size and abs(seconds / size - round(seconds / size)) < 1e-9:
            return f"{int(round(seconds / size))}{unit}"
    return f"{seconds:g}s"


class StoppingSpec(BaseModel):
    """The stopping specification of #224, as accepted by ``POST /api/v1/start``.

    Field names, defaults and validation messages follow the controller
    library, so a specification that validates here validates there.
    """

    model_config = ConfigDict(extra="forbid")

    rel_error: float | None = None
    abs_error: float | None = None
    confidence: float | None = None
    max_trials: int | None = None
    max_time: float | str | None = None
    trials: int | None = None
    goal_queries: list[str] | None = None
    require_goal: bool = False

    @model_validator(mode="before")
    @classmethod
    def _reject_nulls_and_booleans(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        # `{"confidence": null}` is a client bug; absent is the way to take the default.
        if "confidence" in data and data["confidence"] is None:
            raise ValueError("confidence must be a number in (0, 1), not null")
        # Pydantic would coerce True to 1.0/1, which silently means something else.
        for field in ("rel_error", "abs_error", "confidence", "max_trials", "trials", "max_time"):
            if isinstance(data.get(field), bool):
                raise ValueError(f"{field} must not be a boolean")
        return data

    @model_validator(mode="after")
    def _check_rules(self) -> "StoppingSpec":
        if self.rel_error is not None and not (0.0 < self.rel_error < 1.0):
            raise ValueError("rel_error must be greater than 0 and less than 1 (a relative half-width)")
        if self.abs_error is not None and not (0.0 < self.abs_error <= 1.0):
            raise ValueError("abs_error must be greater than 0 and at most 1")
        if self.confidence is not None and not (0.0 < self.confidence < 1.0):
            raise ValueError("confidence must be greater than 0 and less than 1")
        if self.max_trials is not None and self.max_trials < 1:
            raise ValueError("max_trials must be at least 1")
        if self.trials is not None and self.trials < 1:
            raise ValueError("trials must be at least 1")
        if self.trials is not None:
            if self.rel_error is not None or self.abs_error is not None:
                raise ValueError(
                    "trials (a fixed design) cannot be combined with rel_error or abs_error: "
                    "a fixed design does not stop early"
                )
            if self.goal_queries is not None:
                raise ValueError("trials (a fixed design) has no goal, so goal_queries does not apply")
            if self.require_goal:
                raise ValueError("trials (a fixed design) has no goal, so require_goal does not apply")
        if self.goal_queries is not None and not self.goal_queries:
            raise ValueError("goal_queries must name at least one query, or be omitted")
        try:
            parse_duration(self.max_time)
        except StoppingSpecError as exc:
            raise ValueError(str(exc)) from exc
        return self

    # -- resolution --------------------------------------------------------
    def resolve(self) -> "ResolvedStopping":
        """Apply the #224 defaults and return the executable specification.

        A specification always selects a goal or a fixed design: ``-N`` is
        exclusive, and the goal defaults to ``rel_error = 0.01`` and a ten-minute
        limit. ``max_time`` set to ``"none"`` or ``null`` removes the limit but
        keeps the goal, which is what ``fzb run model.pcfb --max-time none`` means
        in #224 and what a UI that cleared its time field sends. An entirely
        unbounded run (no goal, no limits) can only come from the legacy
        vocabulary, which predates the goal default.
        """
        explicit = self.model_fields_set
        confidence = self.confidence if self.confidence is not None else DEFAULT_CONFIDENCE

        if self.trials is not None:
            return ResolvedStopping(
                mode="fixed",
                rel_error=None,
                abs_error=None,
                confidence=confidence,
                max_trials=self.max_trials,
                max_time_s=parse_duration(self.max_time) if "max_time" in explicit else None,
                fixed_trials=self.trials,
                goal_queries=(),
                require_goal=False,
            )

        # A relative goal is the default only when nothing else asks for a bound.
        rel_error = self.rel_error
        if rel_error is None and self.abs_error is None:
            rel_error = DEFAULT_REL_ERROR
        return ResolvedStopping(
            mode="goal",
            rel_error=rel_error,
            abs_error=self.abs_error,
            confidence=confidence,
            max_trials=self.max_trials,
            max_time_s=parse_duration(self.max_time) if "max_time" in explicit else DEFAULT_GOAL_MAX_TIME_S,
            fixed_trials=None,
            goal_queries=tuple(self.goal_queries or ()),
            require_goal=self.require_goal,
        )

    def describe(self) -> dict[str, Any]:
        """The resolved specification as JSON (what the run will do)."""
        return self.resolve().as_dict()


@dataclass(frozen=True, slots=True)
class ResolvedStopping:
    """An executable stopping specification: a goal, limits, or a fixed design."""

    mode: Literal["goal", "fixed", "open"]
    rel_error: float | None
    abs_error: float | None
    confidence: float
    max_trials: int | None
    max_time_s: float | None
    fixed_trials: int | None
    goal_queries: tuple[str, ...] = ()
    require_goal: bool = False
    #: True when the specification was produced by the legacy translation.
    legacy: bool = False
    deprecations: tuple[str, ...] = field(default=())

    @property
    def interval(self) -> str:
        """The interval the run reports. Fixed in #224; zero hits report a CL bound."""
        return "wilson"

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "rel_error": self.rel_error,
            "abs_error": self.abs_error,
            "confidence": self.confidence,
            "max_trials": self.max_trials,
            "max_time": format_duration(self.max_time_s),
            "max_time_s": self.max_time_s,
            "trials": self.fixed_trials,
            "goal_queries": list(self.goal_queries),
            "require_goal": self.require_goal,
            "interval": self.interval,
            "legacy": self.legacy,
        }

    def summary_line(self) -> str:
        """Human-readable one-liner used in run events and the UI status."""
        parts: list[str] = []
        if self.mode == "fixed":
            parts.append(f"fixed design, {self.fixed_trials} trials")
        elif self.mode == "open":
            parts.append("no stopping criterion")
        else:
            goal = []
            if self.rel_error is not None:
                goal.append(f"+/-{self.rel_error * 100:g}%")
            if self.abs_error is not None:
                goal.append(f"abs {self.abs_error:g}")
            parts.append(f"goal {' / '.join(goal)} at {self.confidence * 100:g}%")
        if self.max_trials is not None:
            parts.append(f"max {self.max_trials} trials")
        if self.max_time_s is not None:
            parts.append(f"max {format_duration(self.max_time_s)}")
        return ", ".join(parts)


def check_executable(spec: ResolvedStopping) -> None:
    """Reject the parts of a valid #224 specification phase 1 cannot honour."""
    if spec.goal_queries:
        raise UnsupportedSpecError(
            "goal_queries " + CONTROLLER_REQUIRED + ": phase 1 evaluates the goal on the run's "
            "primary query only, and per-query goals need the controller's policy loop"
        )
    if spec.require_goal:
        raise UnsupportedSpecError(
            "require_goal " + CONTROLLER_REQUIRED + ": use the run result's converged flag and "
            "stop_reason to tell a met goal from a limit"
        )


# ---------------------------------------------------------------------------
#  Legacy options
# ---------------------------------------------------------------------------
#: Request/option keys that belong to the pre-#224 stopping vocabulary.
LEGACY_STOPPING_KEYS: tuple[str, ...] = (
    "num_trials",
    "early_stop",
    "delta",
    "confidence",
    "burn_in",
    "max_passes",
    "policy",
    "stopping_policy",
    "policies",
)

_KNOWN_POLICIES = {"wald", "gbas"}


class LegacyTranslationError(StoppingSpecError):
    """A legacy request whose meaning cannot be expressed faithfully."""


@dataclass(frozen=True, slots=True)
class LegacyTranslation:
    """The specification a legacy request means, plus what changed."""

    spec: ResolvedStopping
    deprecations: tuple[str, ...]
    retained: dict[str, Any] = field(default_factory=dict)


def _legacy_policy(options: Mapping[str, Any]) -> str:
    for key in ("stopping_policy", "policy"):
        raw = options.get(key)
        if raw is not None:
            name = str(raw).strip().lower()
            if name not in _KNOWN_POLICIES:
                raise LegacyTranslationError(
                    f"{key}={raw!r} is not a stopping policy; the legacy values are "
                    f"{sorted(_KNOWN_POLICIES)}, and the controller library owns new ones (#224)"
                )
            return name
    policies = options.get("policies")
    if policies:
        names = [str(item).strip().lower() for item in policies]
        for name in names:
            if name not in _KNOWN_POLICIES:
                raise LegacyTranslationError(
                    f"policies contains {name!r}; the legacy values are {sorted(_KNOWN_POLICIES)}"
                )
        return names[-1]
    return "wald"


def translate_legacy(options: Mapping[str, Any]) -> LegacyTranslation:
    """Map the legacy option names onto the #224 stopping specification.

    Every transformation is exact or the request is rejected; nothing is
    reinterpreted silently. The rules are documented in ``tools/server/README.md``.
    """
    deprecations: list[str] = []
    retained: dict[str, Any] = {}

    delta = options.get("delta")
    confidence = options.get("confidence")
    num_trials = options.get("num_trials")
    early_stop = options.get("early_stop")
    burn_in = options.get("burn_in")

    if delta is not None:
        try:
            delta_value = float(delta)
        except (TypeError, ValueError) as exc:
            raise LegacyTranslationError(f"delta={delta!r} is not a number") from exc
        if not (0.0 < delta_value < 1.0):
            raise LegacyTranslationError("delta must be greater than 0 and less than 1")
        deprecations.append(
            "delta is deprecated: it is the goal's rel_error (#224), but the goal is now "
            "evaluated on the Wilson interval rather than the Wald interval"
        )
        rel_error: float | None = delta_value
    else:
        rel_error = None

    if confidence is not None:
        deprecations.append(
            "confidence is deprecated: it is the stopping specification's confidence (#224)"
        )

    policy = _legacy_policy(options)
    if policy == "gbas":
        deprecations.append(
            "policy=gbas is deprecated: Huber's exact relative-error control becomes a controller "
            "policy (#224); this run uses the Wilson goal at the same delta and confidence"
        )
    else:
        deprecations.append(
            "policy=wald is deprecated: the Wald interval is replaced by the Wilson interval "
            "(#224), which has non-zero width at zero hits"
        )

    if burn_in is not None and float(burn_in) > 0.0:
        raise LegacyTranslationError(
            f"burn_in={burn_in} has no faithful equivalent: burn-in is counted in trials, so its "
            "meaning depends on p, while the #224 goal implies a hit count (about (z/rel_error)^2 "
            "hits whatever p is). Drop burn_in, or express the intent as max_trials. "
            + CONTROLLER_REQUIRED + " for a burn-in in hits"
        )

    if num_trials is not None:
        try:
            budget = float(num_trials)
        except (TypeError, ValueError) as exc:
            raise LegacyTranslationError(f"num_trials={num_trials!r} is not a number") from exc
        if budget < 0.0 or not math.isfinite(budget):
            raise LegacyTranslationError("num_trials must be zero (no limit) or a positive trial budget")
        if budget != int(budget) and budget < 1.0:
            raise LegacyTranslationError("num_trials must be a whole number of trials")
        budget = int(budget)
    else:
        budget = 0

    wants_early_stop = _truthy(early_stop) if early_stop is not None else budget == 0

    if max_passes := options.get("max_passes"):
        retained["max_passes"] = int(float(max_passes))
        deprecations.append(
            "max_passes is server-only and deprecated: it bounds device passes, not trials, "
            "so it is not part of the stopping specification (#224)"
        )

    def _out(spec: ResolvedStopping) -> LegacyTranslation:
        return LegacyTranslation(
            spec=replace(spec, legacy=True, deprecations=tuple(deprecations)),
            deprecations=tuple(deprecations),
            retained=retained,
        )

    if budget <= 0 and not wants_early_stop:
        spec = ResolvedStopping(
            mode="open",
            rel_error=None,
            abs_error=None,
            confidence=_confidence_value(confidence),
            max_trials=None,
            max_time_s=None,
            fixed_trials=None,
        )
        deprecations.append(
            "num_trials=0 with early_stop=false means no stopping criterion: the run continues "
            "until it is stopped. Send max_time=none and no goal to say so explicitly"
        )
        return _out(spec)

    if budget > 0 and not wants_early_stop:
        spec = ResolvedStopping(
            mode="fixed",
            rel_error=None,
            abs_error=None,
            confidence=_confidence_value(confidence),
            max_trials=None,
            max_time_s=None,
            fixed_trials=budget,
        )
        deprecations.append(
            "num_trials with early_stop=false is a fixed design: it now reports stop_reason "
            "'trials' and commits whole passes (see README: pass quantisation)"
        )
        return _out(spec)

    spec = ResolvedStopping(
        mode="goal",
        rel_error=rel_error if rel_error is not None else DEFAULT_REL_ERROR,
        abs_error=None,
        confidence=_confidence_value(confidence),
        max_trials=budget or None,
        max_time_s=None if budget <= 0 else DEFAULT_GOAL_MAX_TIME_S,
        fixed_trials=None,
    )
    if budget > 0:
        deprecations.append(
            "num_trials with early_stop=true is a trial limit (max_trials): the goal stops the "
            "run earlier and the limit stops it later"
        )
    else:
        deprecations.append(
            "num_trials=0 with early_stop=true is a goal with no limits: it keeps its unbounded "
            "behaviour instead of taking the ten-minute goal default"
        )
    return _out(spec)


def _confidence_value(confidence: Any) -> float:
    if confidence is None:
        return DEFAULT_CONFIDENCE
    try:
        value = float(confidence)
    except (TypeError, ValueError) as exc:
        raise LegacyTranslationError(f"confidence={confidence!r} is not a number") from exc
    if not (0.0 < value < 1.0):
        raise LegacyTranslationError("confidence must be greater than 0 and less than 1")
    return value


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def sampler_budget(spec: ResolvedStopping) -> dict[str, Any]:
    """Engine sizing for ``spec``: a budget sizes passes to cover it exactly once."""
    budget = spec.fixed_trials if spec.fixed_trials is not None else spec.max_trials
    return {"num_trials": budget} if budget else {}


__all__ = [
    "CONTROLLER_REQUIRED",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_GOAL_MAX_TIME_S",
    "DEFAULT_REL_ERROR",
    "LEGACY_STOPPING_KEYS",
    "LegacyTranslation",
    "LegacyTranslationError",
    "ResolvedStopping",
    "StoppingSpec",
    "StoppingSpecError",
    "UnsupportedSpecError",
    "check_executable",
    "format_duration",
    "parse_duration",
    "sampler_budget",
    "translate_legacy",
]
