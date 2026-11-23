/**
 * The stopping specification the server accepts as `stopping` (#224).
 *
 * Three questions stay separate, as on `fzb run`:
 *  - goal: how precise must the answer be (`rel_error`, the reported interval's
 *    half-width relative to the estimate; `abs_error` as a floor for tiny
 *    probabilities);
 *  - limits: how much may the run spend (`max_trials`, `max_time`);
 *  - design: a fixed trial count (`trials`), which has no early stop.
 *
 * The server validates the same model (`fuzzybits_server.convergence.stopping`)
 * and rejects, with a 422, anything the current loop cannot honour faithfully.
 * This module is deliberately dependency-free so the form's semantics can be
 * unit-tested with `node --test`.
 */

export type StoppingPayload = {
  rel_error?: number;
  abs_error?: number;
  confidence?: number;
  max_trials?: number;
  /** Seconds, or "none" for no time limit. */
  max_time?: number | "none";
  trials?: number;
};

/** What the server reports in a control response and in the run status. */
export type StoppingSummary = {
  mode: "goal" | "fixed" | "open";
  rel_error: number | null;
  abs_error: number | null;
  confidence: number;
  max_trials: number | null;
  max_time: string;
  max_time_s: number | null;
  trials: number | null;
  goal_queries: string[];
  require_goal: boolean;
  interval: string;
  legacy: boolean;
  deprecations: string[];
};

export type StoppingMode = "goal" | "fixed";

export const DEFAULT_PRECISION_PERCENT = 1;
export const MIN_PRECISION_PERCENT = 0.01;
export const MAX_PRECISION_PERCENT = 50;

export const DEFAULT_CONFIDENCE_PERCENT = 95;
export const DEFAULT_TIME_LIMIT_MINUTES = 10;

export const TIME_LIMIT_UNITS = ["s", "m", "h"] as const;
export type TimeLimitUnit = (typeof TIME_LIMIT_UNITS)[number];
export const TIME_LIMIT_UNIT_SECONDS: Record<TimeLimitUnit, number> = { s: 1, m: 60, h: 3600 };

/**
 * Stop reasons the run reports (#224), plus the legacy loop's own names so a
 * frame from a run started before the settings change still renders.
 */
export const STOP_REASON_LABELS: Record<string, string> = {
  goal: "Goal met",
  "max-trials": "Trial limit",
  "max-time": "Time limit",
  trials: "Fixed design complete",
  interrupted: "Interrupted",
  converged: "Converged",
  max_trials: "Trial limit",
  max_passes: "Pass limit",
};

export function stopReasonLabel(reason: string | null | undefined): string {
  if (!reason) return "Running";
  return STOP_REASON_LABELS[reason] ?? reason;
}

/** Whether a stop reason means the run reached its target, not a limit. */
export function stopReasonIsSuccess(reason: string | null | undefined): boolean {
  return reason === "goal" || reason === "trials" || reason === "converged";
}

// ---------------------------------------------------------------------------
//  Precision (%)
// ---------------------------------------------------------------------------
export function clampPrecisionPercent(percent: number): number {
  if (!Number.isFinite(percent)) return DEFAULT_PRECISION_PERCENT;
  const clamped = Math.min(MAX_PRECISION_PERCENT, Math.max(MIN_PRECISION_PERCENT, percent));
  return Number(clamped.toFixed(4));
}

export function precisionPercentToRelError(percent: number): number {
  return clampPrecisionPercent(percent) / 100;
}

export function relErrorToPrecisionPercent(relError: number): number {
  return clampPrecisionPercent(relError * 100);
}

export function formatPrecisionPercent(percent: number): string {
  return `±${formatTrimmed(clampPrecisionPercent(percent))}%`;
}

// ---------------------------------------------------------------------------
//  Time limits
// ---------------------------------------------------------------------------
/** Seconds -> the value and unit the time-limit control shows. */
export function splitTimeLimit(seconds: number): { value: number; unit: TimeLimitUnit } {
  for (const unit of ["h", "m"] as const) {
    const factor = TIME_LIMIT_UNIT_SECONDS[unit];
    if (seconds >= factor && Number.isInteger(seconds / factor)) {
      return { value: seconds / factor, unit };
    }
  }
  return { value: seconds, unit: "s" };
}

export function joinTimeLimit(value: number, unit: TimeLimitUnit): number {
  return value * TIME_LIMIT_UNIT_SECONDS[unit];
}

export function formatTimeLimit(seconds: number | null | "none" | undefined): string {
  if (seconds === null || seconds === undefined || seconds === "none") return "no limit";
  const { value, unit } = splitTimeLimit(seconds);
  return `${formatTrimmed(value)}${unit}`;
}

// ---------------------------------------------------------------------------
//  The payload and its description
// ---------------------------------------------------------------------------
export type RunSettingsInput = {
  mode: StoppingMode;
  precisionPercent: number;
  /** Two-sided confidence level (%), for the goal and the report. */
  confidencePercent: number;
  /** Wall-clock limit in seconds; null means none. */
  timeLimitSeconds: number | null;
  /** Trial limit; 0 means none. */
  maxTrials: number;
  /** The fixed design's trial count; ignored in goal mode. */
  fixedTrials: number;
};

/**
 * Build the `stopping` object from the form state.
 *
 * A fixed design sends `trials` and no goal, because `-N` is exclusive with one;
 * a goal sends the precision, the confidence and the limits. `max_time` is always
 * explicit, so "no limit" is unambiguous rather than an omitted default.
 */
export function buildStoppingPayload(input: RunSettingsInput): StoppingPayload {
  const maxTime: number | "none" = input.timeLimitSeconds === null ? "none" : input.timeLimitSeconds;
  if (input.mode === "fixed") {
    return {
      trials: Math.max(1, Math.round(input.fixedTrials)),
      ...(input.timeLimitSeconds === null ? {} : { max_time: maxTime }),
      ...(Number.isFinite(input.confidencePercent) ? { confidence: input.confidencePercent / 100 } : {}),
    };
  }
  const payload: StoppingPayload = {
    rel_error: precisionPercentToRelError(input.precisionPercent),
    confidence: (Number.isFinite(input.confidencePercent) ? input.confidencePercent : DEFAULT_CONFIDENCE_PERCENT) / 100,
    max_time: maxTime,
  };
  if (input.maxTrials > 0) payload.max_trials = Math.round(input.maxTrials);
  return payload;
}

/** One-line description of what a run will do, mirroring the server's summary. */
export function describeStopping(payload: StoppingPayload): string {
  const parts: string[] = [];
  if (payload.trials !== undefined) {
    parts.push(`fixed design, ${formatCount(payload.trials)} trials`);
  } else {
    const goal: string[] = [];
    if (payload.rel_error !== undefined) {
      goal.push(formatPrecisionPercent(relErrorToPrecisionPercent(payload.rel_error)));
    }
    if (payload.abs_error !== undefined) goal.push(`abs ${payload.abs_error}`);
    const level = payload.confidence !== undefined ? `${formatTrimmed(payload.confidence * 100)}%` : "95%";
    parts.push(`goal ${goal.join(" / ")} at ${level}`);
  }
  if (payload.max_trials !== undefined) parts.push(`max ${formatCount(payload.max_trials)} trials`);
  if (payload.max_time !== undefined) parts.push(`max ${formatTimeLimit(payload.max_time)}`);
  return parts.join(", ");
}

/** The budget the sample-shape view should be sized for. */
export function trialBudget(input: RunSettingsInput): number {
  if (input.mode === "fixed") return Math.max(1, Math.round(input.fixedTrials));
  return input.maxTrials > 0 ? Math.round(input.maxTrials) : 0;
}

export const STOPPING_FIELD_HINT = [
  "A run either meets a precision goal or runs a fixed number of trials.",
  "Goal: stop as soon as the reported Wilson interval's half-width is within ±p of the estimate, at the chosen confidence, or when a limit is reached.",
  "Fixed: run exactly that many trials with no early stop; use it to compare runs.",
  "A relative goal cannot be met at zero hits, so keep a trial or time limit on so the run still ends.",
].join(" ");

function formatTrimmed(value: number): string {
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function formatCount(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}
