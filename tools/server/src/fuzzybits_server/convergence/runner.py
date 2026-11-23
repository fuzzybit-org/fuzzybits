"""The pass -> tally -> update -> emit loop and its configuration.

:class:`ConvergenceRunner` ties the layers together:

1. pull one :class:`PassObservation` from a :class:`Sampler`,
2. fold it into :class:`RunState`,
3. evaluate every attached :class:`ConvergencePolicy` and :class:`Metric`,
4. assemble a flat progress frame, and
5. decide whether to stop (burn-in, early-stop on the chosen policy, or a
   trial/pass budget).

The progress frame keeps the field names the existing web client consumes
(``mean``, ``epsilon``, ``target_epsilon``, ``epsilon_converged``,
``p01/05/95/99``, ``info_bits_*`` …) filled from the *primary* policy (GBAS
when it is attached, otherwise Wald), and adds every policy/metric under its own
namespace (``wald_*``, ``gbas_*``, ``info_bits_*`` …) so additional streams are
available without breaking the UI.

:class:`ConvergenceConfig` carries the host-side knobs and is built from the
existing server options dict via :meth:`ConvergenceConfig.from_options`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

from .metrics import Metric, default_metrics
from .policies import ConvergencePolicy, PolicyVerdict, make_policy
from .sampling import Sampler, make_sampler
from .special import normal_quantile_two_sided
from .stopping import LEGACY_STOPPING_KEYS, ResolvedStopping, sampler_budget
from .tally import RunContext, RunState
from .wilson import WilsonGoalPolicy

ProgressCallback = Callable[[dict[str, Any]], None]

#: Stopping-specification option keys that are not legacy names but are still
#: consumed by the specification rather than by :meth:`ConvergenceConfig.from_options`.
_STOPPING_EXTRA_KEYS: frozenset[str] = frozenset(
    {"goal_rel_error", "goal_abs_error", "fixed_trials", "max_time_s", "stop_reason_style"}
)

# Default primary policy: fills the canonical (un-namespaced) frontend fields.
# A goal run's Wilson policy outranks GBAS: its interval is the one the goal is
# evaluated on, so it is also the one the client must show.
PRIMARY_POLICY = "gbas"
PRIMARY_ORDER = ("wilson", "gbas")

#: Stop reasons in the controller's vocabulary (#224). The legacy host loop
#: reports its own names ("converged", "max_trials", "max_passes") so that a
#: request that does not opt into the stopping specification keeps the wire
#: behaviour it had before #225; see ``tools/server/README.md``.
STOP_GOAL = "goal"
STOP_MAX_TRIALS = "max-trials"
STOP_MAX_TIME = "max-time"
STOP_TRIALS = "trials"
STOP_INTERRUPTED = "interrupted"

INTERVAL_WILSON = "wilson"
INTERVAL_WALD = "wald"


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    f = _as_float(value)
    return int(round(f)) if f is not None else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _sanitize(value: Any) -> Any:
    """Make a progress value safe for strict JSON (browsers reject NaN/Infinity)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


@dataclass(frozen=True)
class ConvergenceConfig:
    """Host-side convergence configuration (independent of the native settings).

    Note: intentionally *not* ``slots=True`` -- :meth:`from_options` reads field
    defaults via ``cls.<field>`` (which slots would shadow with descriptors).
    """

    delta: float = 1.0e-3
    confidence: float = 0.99
    burn_in: int = 0
    max_trials: int = 0  # 0 -> unlimited
    max_passes: int = 0  # 0 -> derive from max_trials, else unbounded
    early_stop: bool = False
    oracle_p: float | None = None
    policies: tuple[str, ...] = ("wald",)
    stopping_policy: str = "wald"
    event_ids: tuple[int, ...] = ()
    emit_min_interval_s: float = 1.0 / 60.0
    stream_nodes: bool = False
    # Device passes per sampler step; the host reads tallies and checks
    # convergence once per step.
    passes_per_step: int = 1
    # --- #224 stopping specification -------------------------------------
    # Goal evaluated on the Wilson interval with the `max(eps*p_hat, a)` rule.
    goal_rel_error: float | None = None
    goal_abs_error: float | None = None
    # Exactly this many committed trials, no early stop (`-N`).
    fixed_trials: int | None = None
    # Wall-clock limit in seconds.
    max_time_s: float | None = None
    # Report the controller's stop reasons and treat goal/limits/fixed design
    # as the stopping rule. False keeps the pre-#225 loop verbatim.
    stop_reason_style: str = "legacy"

    @property
    def controller_mode(self) -> bool:
        """Whether this config executes a #224 stopping specification."""
        return self.stop_reason_style == "controller"

    @classmethod
    def from_stopping(cls, spec: ResolvedStopping, options: dict[str, Any] | None = None) -> "ConvergenceConfig":
        """Build the config that executes a resolved #224 stopping specification.

        ``options`` may still carry the non-stopping knobs (oracle, watch,
        passes_per_step, ...); every stopping key in it is ignored, so the
        specification is the only source of the stopping rule.
        """
        base = cls.from_options(
            {
                key: value
                for key, value in (options or {}).items()
                if key not in LEGACY_STOPPING_KEYS and key not in _STOPPING_EXTRA_KEYS
            }
        )
        policies: tuple[str, ...] = ("wald", "wilson")
        return replace(
            base,
            confidence=spec.confidence,
            goal_rel_error=spec.rel_error,
            goal_abs_error=spec.abs_error,
            fixed_trials=spec.fixed_trials,
            max_trials=spec.max_trials or 0,
            max_time_s=spec.max_time_s,
            early_stop=spec.mode == "goal",
            stopping_policy="wilson",
            policies=policies,
            stop_reason_style="controller",
        )

    @classmethod
    def from_options(cls, options: dict[str, Any] | None) -> "ConvergenceConfig":
        """Build a config from the server's CLI-like options mapping."""
        options = options or {}
        delta = _as_float(options.get("delta"))
        confidence = _as_float(options.get("confidence"))
        burn_in = _as_int(options.get("burn_in"))
        max_trials = _as_int(options.get("num_trials"))
        oracle = _as_float(options.get("oracle"))
        policy = options.get("policy")
        stopping = str(policy).strip().lower() if policy else cls.stopping_policy
        # Always evaluate Wald so its stats stream regardless of which policy
        # gates the stop.
        policies: tuple[str, ...] = ("wald",)
        if stopping not in policies:
            policies = (*policies, stopping)

        emit_interval = _as_float(options.get("emit_min_interval_s"))
        event_ids = tuple(int(i) for i in options.get("convergence_event_ids", ()) or ())

        return cls(
            delta=delta if delta is not None and delta > 0 else cls.delta,
            confidence=confidence if confidence is not None else cls.confidence,
            burn_in=max(0, burn_in or 0),
            max_trials=max(0, max_trials or 0),
            max_passes=max(0, _as_int(options.get("max_passes")) or 0),
            early_stop=_as_bool(options.get("early_stop", False)),
            oracle_p=oracle if (oracle is not None and oracle >= 0.0) else None,
            policies=policies,
            stopping_policy=stopping,
            event_ids=event_ids,
            emit_min_interval_s=emit_interval if emit_interval is not None else cls.emit_min_interval_s,
            stream_nodes=_as_bool(options.get("stream_nodes", False)),
            passes_per_step=max(1, _as_int(options.get("passes_per_step")) or 1),
        )


@dataclass(slots=True)
class RunResult:
    """Outcome of a completed (or stopped) convergence run."""

    state: RunState
    final_progress: dict[str, Any]
    stop_reason: str | None
    converged: bool
    elapsed_s: float

    @property
    def summary(self) -> dict[str, Any]:
        """A native-``run_probability``-compatible result dict for the worker."""
        target = self.state.target
        mean = target.mean

        # If the stopping policy publishes its own point estimate (e.g. GBAS's
        # Gamma-corrected ``(k-1)/R``, whose relative error is exactly known),
        # prefer it as the authoritative result over the raw sample proportion.
        stopping = self.final_progress.get("stopping_policy")
        if stopping and self.converged:
            pol_mean = self.final_progress.get(f"{stopping}_mean")
            if isinstance(pol_mean, (int, float)) and math.isfinite(float(pol_mean)):
                mean = float(pol_mean)

        result: dict[str, Any] = {
            "analysis_time": self.elapsed_s,
            "warnings": "",
            "probabilities": [{"p_total": mean, "analysis_time": self.elapsed_s}],
            "engine": "python",
            "converged": self.converged,
            "stop_reason": self.stop_reason,
            "mean": mean,
            "sample_mean": target.mean,
            "trials": target.total_bits,
            "passes": self.state.pass_count,
            "stopping_policy": stopping,
        }

        # Carry the stopping policy's confidence interval when it provides one.
        ci_low = self.final_progress.get(f"{stopping}_ci_low") if stopping else None
        ci_high = self.final_progress.get(f"{stopping}_ci_high") if stopping else None
        if isinstance(ci_low, (int, float)) and isinstance(ci_high, (int, float)):
            result["ci_low"] = float(ci_low)
            result["ci_high"] = float(ci_high)
        return result


def _make_policy(name: str, config: "ConvergenceConfig") -> ConvergencePolicy:
    """Instantiate the configured policy, binding a goal's tolerance to it."""
    if name == WilsonGoalPolicy.name:
        return WilsonGoalPolicy(rel_error=config.goal_rel_error, abs_error=config.goal_abs_error)
    return make_policy(name)


class ConvergenceRunner:
    """Drives the host-side convergence loop for a single sampler."""

    def __init__(
        self,
        sampler: Sampler,
        config: ConvergenceConfig,
        *,
        policies: Sequence[ConvergencePolicy] | None = None,
        metrics: Sequence[Metric] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.sampler = sampler
        self.config = config
        self.on_progress = on_progress
        self.policies: list[ConvergencePolicy] = (
            list(policies)
            if policies is not None
            else [_make_policy(name, config) for name in config.policies]
        )
        self.metrics: list[Metric] = (
            list(metrics) if metrics is not None else default_metrics()
        )
        self.state = RunState(target_index=sampler.target_index)

        z = normal_quantile_two_sided(config.confidence)
        self.ctx = RunContext(
            confidence=config.confidence,
            z=z,
            delta=config.delta,
            trials_per_pass=sampler.shape.trials_per_pass,
            target_index=sampler.target_index,
            oracle_p=config.oracle_p,
        )

        by_name = {policy.name: policy for policy in self.policies}
        self._primary_name = next(
            (name for name in PRIMARY_ORDER if name in by_name), self.policies[0].name
        )
        self._stopping_name = (
            config.stopping_policy if config.stopping_policy in by_name else self._primary_name
        )

    # -- public API --------------------------------------------------------
    def run(self) -> RunResult:
        """Run until a stop condition fires; returns the final :class:`RunResult`."""
        cfg = self.config
        tpp = max(1, self.sampler.shape.trials_per_pass)
        max_passes = cfg.max_passes
        if not max_passes and cfg.max_trials:
            max_passes = math.ceil(cfg.max_trials / tpp)
        if not max_passes and cfg.fixed_trials:
            # The engine rounds the budget up to whole bitpacks and caps the pass
            # size, so its own iteration count is the exact pass budget for a
            # fixed design; recomputing it here would round twice.
            max_passes = self.sampler.shape.total_iterations or math.ceil(cfg.fixed_trials / tpp)

        start = time.monotonic()
        last_emit = -math.inf
        last_phase: str | None = None
        final_progress: dict[str, Any] = {}
        stop_reason: str | None = None
        converged = False

        while True:
            passes = cfg.passes_per_step
            if max_passes:
                passes = max(1, min(passes, max_passes - self.state.pass_count))
            observation = self.sampler.step(passes)
            self.state.update(observation)
            target = self.state.target

            phase = "burn_in" if target.total_bits < cfg.burn_in else "sampling"
            verdicts = {policy.name: policy.evaluate(target, self.ctx) for policy in self.policies}

            stopping_converged = verdicts[self._stopping_name].converged and phase == "sampling"
            elapsed = time.monotonic() - start
            stop_reason = self._stop_reason(
                phase=phase,
                stopping_converged=stopping_converged,
                total_bits=target.total_bits,
                pass_count=self.state.pass_count,
                max_passes=max_passes,
                elapsed_s=elapsed,
            )
            converged = stopping_converged
            stopping = stop_reason is not None

            progress = self._build_progress(phase, verdicts)
            progress["converged"] = stopping_converged
            progress["elapsed_s"] = elapsed
            progress["eta_seconds"] = self._eta_seconds(target.total_bits, verdicts, elapsed)
            if stopping:
                progress["stop_reason"] = stop_reason
            final_progress = progress

            force = stopping or last_phase != phase or self.state.pass_count == 1
            if self.on_progress is not None and (
                force or (observation.wall_time - last_emit) >= cfg.emit_min_interval_s
            ):
                self.on_progress(progress)
                last_emit = observation.wall_time
            last_phase = phase

            if stopping:
                break

        return RunResult(
            state=self.state,
            final_progress=final_progress,
            stop_reason=stop_reason,
            converged=converged,
            elapsed_s=time.monotonic() - start,
        )

    # -- internals ---------------------------------------------------------
    def _stop_reason(
        self,
        *,
        phase: str,
        stopping_converged: bool,
        total_bits: int,
        pass_count: int,
        max_passes: int,
        elapsed_s: float = 0.0,
    ) -> str | None:
        if self.config.controller_mode:
            return self._controller_stop_reason(
                stopping_converged=stopping_converged,
                total_bits=total_bits,
                elapsed_s=elapsed_s,
            )
        if self.config.max_trials and total_bits >= self.config.max_trials:
            return "max_trials"
        if max_passes and pass_count >= max_passes:
            return "max_passes"
        if phase == "sampling" and self.config.early_stop and stopping_converged:
            return "converged"
        return None

    def _controller_stop_reason(
        self,
        *,
        stopping_converged: bool,
        total_bits: int,
        elapsed_s: float,
    ) -> str | None:
        """Stop reasons in the controller's vocabulary (#224).

        A fixed design runs to its trial count; a goal stops at the end of the
        first committed pass at which every goal query meets it; either stops
        earlier on a limit. The goal is checked before the limits so a run that
        reaches its trial limit exactly when it converges reports ``goal``.
        """
        cfg = self.config
        if cfg.fixed_trials and total_bits >= cfg.fixed_trials:
            return STOP_TRIALS
        if stopping_converged:
            return STOP_GOAL
        if cfg.max_trials and total_bits >= cfg.max_trials:
            return STOP_MAX_TRIALS
        if cfg.max_time_s is not None and elapsed_s >= cfg.max_time_s:
            return STOP_MAX_TIME
        return None

    def _eta_seconds(
        self,
        total_bits: int,
        verdicts: dict[str, PolicyVerdict],
        elapsed_s: float,
    ) -> float | None:
        """Seconds to the projected trial count at the observed rate, or ``None``."""
        projected = verdicts[self._stopping_name].projected_trials
        if not projected or not math.isfinite(projected) or projected <= total_bits:
            return None
        if elapsed_s <= 0.0 or total_bits <= 0:
            return None
        rate = total_bits / elapsed_s
        return (projected - total_bits) / rate if rate > 0.0 else None

    def _build_progress(
        self, phase: str, verdicts: dict[str, PolicyVerdict]
    ) -> dict[str, Any]:
        target = self.state.target
        tpp = max(1, self.sampler.shape.trials_per_pass)
        total_bits = target.total_bits

        progress: dict[str, Any] = {
            "phase": phase,
            "pass_index": self.state.pass_count,
            "iterations": self.state.pass_count,
            "trials": total_bits,
            "trials_per_pass": tpp,
        }

        # Auxiliary metrics (point estimate, info gain, diagnostics).
        for metric in self.metrics:
            progress.update(metric.compute(target, self.ctx))

        # Primary policy fills the canonical (un-namespaced) fields.
        primary = verdicts[self._primary_name]
        for key, value in primary.metrics.items():
            if value is not None:
                progress[key] = value
        progress["epsilon"] = primary.epsilon
        progress["target_epsilon"] = primary.target_epsilon
        progress["epsilon_converged"] = primary.converged
        # GBAS does not emit Wald percentile bands; keep them on the canonical
        # frame from Wald when both policies are attached.
        if self._primary_name != "wald" and "wald" in verdicts:
            wald_metrics = verdicts["wald"].metrics
            for key in ("std_err", "p01", "p05", "p95", "p99"):
                if progress.get(key) is None:
                    value = wald_metrics.get(key)
                    if value is not None:
                        progress[key] = value
        projected = primary.projected_trials
        progress["projected_trials"] = projected
        remaining = max(0, projected - total_bits)
        progress["remaining_trials"] = remaining
        # An infinite projection (a relative goal at zero hits has no finite trial
        # count that meets it) is reported as absent, like every non-finite value.
        progress["projected_iterations"] = math.ceil(projected / tpp) if math.isfinite(projected) and projected else 0
        progress["remaining_iterations"] = math.ceil(remaining / tpp) if math.isfinite(remaining) and remaining else 0

        # Every policy is emitted under its namespace so multiple convergence
        # criteria stream simultaneously (wald_*, gbas_*, ...).  Core verdict
        # fields are written last so they win over any duplicated metric keys.
        for name, verdict in verdicts.items():
            for key, value in verdict.metrics.items():
                progress[f"{name}_{key}"] = value
            progress[f"{name}_converged"] = verdict.converged
            progress[f"{name}_epsilon"] = verdict.epsilon
            progress[f"{name}_target_epsilon"] = verdict.target_epsilon
            progress[f"{name}_projected_trials"] = verdict.projected_trials

        progress["primary_policy"] = self._primary_name
        progress["stopping_policy"] = self._stopping_name
        if self.config.controller_mode:
            # Additive fields #224 defines for the controller's snapshots. Absent
            # from a legacy run, so old clients see the frames they always saw.
            progress["interval"] = INTERVAL_WILSON if self._primary_name == "wilson" else INTERVAL_WALD
            progress["stop_reason_style"] = "controller"
            if progress.get("target_rel_half_width") is None and self.config.goal_rel_error is not None:
                progress["target_rel_half_width"] = self.config.goal_rel_error
            if self.config.fixed_trials:
                progress["target_trials"] = self.config.fixed_trials
            elif self.config.max_trials:
                progress["target_trials"] = self.config.max_trials

        if self.config.stream_nodes and len(self.state.nodes) > 1:
            progress["nodes"] = {
                str(index): {"mean": node.mean, "trials": node.total_bits}
                for index, node in self.state.nodes.items()
            }

        return {key: _sanitize(value) for key, value in progress.items()}


def run_calculation(
    input_files: list[str],
    options: dict[str, Any] | None,
    on_progress: ProgressCallback | None = None,
    *,
    pcfb: bytes | None = None,
    pcfb_path: str | None = None,
    stopping: ResolvedStopping | None = None,
) -> dict[str, Any]:
    """Compile, sample, and converge entirely on the host; return a result dict.

    Default engine is ``.pcfb`` / ``fuzzybits.Session``. Set
    ``FUZZYBITS_SERVER_ENGINE=legacy`` for the in-memory ``run_compile`` path.
    The native engine is used *only* to produce raw per-pass popcounts; all
    convergence and statistics are computed here.

    ``stopping`` executes a #224 stopping specification. Without it the legacy
    option vocabulary in ``options`` drives the loop, unchanged.
    """
    config = (
        ConvergenceConfig.from_stopping(stopping, options) if stopping is not None
        else ConvergenceConfig.from_options(options)
    )
    sampler_options = dict(options or {})
    if stopping is not None:
        # The engine is sized from the specification, not from the legacy keys.
        for key in LEGACY_STOPPING_KEYS:
            sampler_options.pop(key, None)
        sampler_options.update(sampler_budget(stopping))
    sampler = make_sampler(
        input_files,
        sampler_options,
        event_ids=list(config.event_ids) or None,
        pcfb=pcfb,
        pcfb_path=pcfb_path,
    )
    try:
        runner = ConvergenceRunner(sampler, config, on_progress=on_progress)
        result = runner.run()
    finally:
        sampler.close()
    summary = result.summary
    if stopping is not None:
        block = stopping.as_dict()
        block["deprecations"] = list(stopping.deprecations)
        summary["stopping"] = block
        summary["stopping_summary"] = stopping.summary_line()
        summary["interval"] = INTERVAL_WILSON
    return summary


__all__ = [
    "ConvergenceConfig",
    "ConvergenceRunner",
    "RunResult",
    "ProgressCallback",
    "STOP_GOAL",
    "STOP_INTERRUPTED",
    "STOP_MAX_TIME",
    "STOP_MAX_TRIALS",
    "STOP_TRIALS",
    "run_calculation",
]
