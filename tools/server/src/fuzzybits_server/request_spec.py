"""Request-model resolution for the run endpoints.

``POST /api/v1/start`` and ``POST /api/v1/reset`` accept two vocabularies:

* the #224 stopping specification (``stopping``), which is the supported one;
* the legacy flat options (``num_trials``, ``early_stop``, ``delta``,
  ``confidence``, ``burn_in``, ``policy``), accepted for one release with a
  deprecation warning, translated by
  :func:`fuzzybits_server.convergence.stopping.translate_legacy`.

A request may use either, not both. A translated legacy request is executed by
the same code path as an explicit specification, so there is one implementation
of the stopping rule and the mapping is exercised on every legacy call.

A specification the host-side loop cannot honour faithfully is rejected with
HTTP 422 and a message naming the controller library (#224); nothing is
approximated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from fuzzybits_server.convergence.stopping import (
    LEGACY_STOPPING_KEYS,
    LegacyTranslation,
    ResolvedStopping,
    StoppingSpec,
    StoppingSpecError,
    UnsupportedSpecError,
    check_executable,
    translate_legacy,
)

#: Legacy keys that carry a stopping meaning. Anything else in the request is a
#: compile or sampler knob and passes through untouched.
LEGACY_KEYS = frozenset(LEGACY_STOPPING_KEYS)


class RequestSpecError(StoppingSpecError):
    """A request the API rejects; ``status_code`` says how."""

    def __init__(self, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class ResolvedRequest:
    """A request's stopping specification plus what the client should be told."""

    stopping: ResolvedStopping
    legacy: bool
    deprecations: tuple[str, ...] = ()
    retained: dict[str, Any] = field(default_factory=dict)

    @property
    def warnings(self) -> list[str]:
        return list(self.deprecations)

    def describe(self) -> dict[str, Any]:
        """The ``stopping`` block returned in control responses and run status."""
        payload = self.stopping.as_dict()
        if self.deprecations:
            payload["deprecations"] = list(self.deprecations)
        return payload


def _has_legacy_stopping(body: Mapping[str, Any]) -> bool:
    return any(key in body and body[key] is not None for key in LEGACY_KEYS)


def _stopping_payload(body: Mapping[str, Any]) -> Any:
    raw = body.get("stopping")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RequestSpecError(f"stopping is not valid JSON: {exc.msg}") from exc
    return raw


def resolve_request(body: Mapping[str, Any] | None) -> ResolvedRequest | None:
    """Resolve a control request body into a stopping specification.

    Returns ``None`` when the body names no stopping intent at all, in which
    case the caller keeps the configured defaults (the legacy path, unchanged).
    """
    if not body:
        return None
    has_spec = body.get("stopping") is not None
    has_legacy = _has_legacy_stopping(body)
    if has_spec and has_legacy:
        legacy_names = sorted(key for key in LEGACY_KEYS if key in body)
        raise RequestSpecError(
            "send either stopping (the stopping specification) or the legacy options "
            f"{legacy_names}, not both",
            status_code=400,
        )

    if has_spec:
        try:
            spec = StoppingSpec.model_validate(_stopping_payload(body))
        except StoppingSpecError as exc:
            raise RequestSpecError(str(exc)) from exc
        except Exception as exc:  # pydantic.ValidationError and friends
            raise RequestSpecError(_validation_message(exc)) from exc
        resolved = spec.resolve()
        try:
            check_executable(resolved)
        except UnsupportedSpecError as exc:
            raise RequestSpecError(str(exc)) from exc
        return ResolvedRequest(
            stopping=resolved,
            legacy=False,
            deprecations=tuple(resolved.deprecations),
        )

    if not has_legacy:
        return None

    try:
        translation: LegacyTranslation = translate_legacy(body)
    except StoppingSpecError as exc:
        raise RequestSpecError(str(exc)) from exc
    try:
        check_executable(translation.spec)
    except UnsupportedSpecError as exc:
        raise RequestSpecError(str(exc)) from exc
    return ResolvedRequest(
        stopping=translation.spec,
        legacy=True,
        deprecations=translation.deprecations,
        retained=translation.retained,
    )


def _validation_message(exc: Exception) -> str:
    """One line per validation error, naming the field as the client sent it."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        parts: list[str] = []
        for error in errors():
            loc = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
            message = str(error.get("msg", "")).removeprefix("Value error, ")
            parts.append(f"{loc}: {message}" if loc else message)
        if parts:
            return "; ".join(parts)
    return str(exc)


__all__ = ["LEGACY_KEYS", "RequestSpecError", "ResolvedRequest", "resolve_request"]
