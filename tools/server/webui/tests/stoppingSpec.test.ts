/**
 * Unit tests for the run-settings semantics (`npm test`, which runs
 * `node --test`; the model modules are dependency-free so node can load them).
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  buildStoppingPayload,
  clampPrecisionPercent,
  describeStopping,
  formatPrecisionPercent,
  formatTimeLimit,
  joinTimeLimit,
  precisionPercentToRelError,
  relErrorToPrecisionPercent,
  splitTimeLimit,
  stopReasonIsSuccess,
  stopReasonLabel,
  trialBudget,
  DEFAULT_CONFIDENCE_PERCENT,
  DEFAULT_PRECISION_PERCENT,
  DEFAULT_TIME_LIMIT_MINUTES,
  MAX_PRECISION_PERCENT,
  MIN_PRECISION_PERCENT,
  type RunSettingsInput,
} from "../src/app/tabs/playground/controls/stoppingSpec.ts";

const GOAL_DEFAULTS: RunSettingsInput = {
  mode: "goal",
  precisionPercent: DEFAULT_PRECISION_PERCENT,
  confidencePercent: DEFAULT_CONFIDENCE_PERCENT,
  timeLimitSeconds: DEFAULT_TIME_LIMIT_MINUTES * 60,
  maxTrials: 0,
  fixedTrials: 1_000_000_000,
};

test("the default goal is ±1% at 95% with a ten-minute limit", () => {
  const payload = buildStoppingPayload(GOAL_DEFAULTS);
  assert.deepEqual(payload, { rel_error: 0.01, confidence: 0.95, max_time: 600 });
});

test("a precision percentage becomes the relative half-width", () => {
  assert.equal(precisionPercentToRelError(1), 0.01);
  assert.equal(precisionPercentToRelError(0.05), 0.0005);
  assert.equal(relErrorToPrecisionPercent(0.005), 0.5);
  assert.equal(formatPrecisionPercent(1), "±1%");
  assert.equal(formatPrecisionPercent(0.5), "±0.5%");
});

test("precision is clamped to the supported range", () => {
  assert.equal(clampPrecisionPercent(MIN_PRECISION_PERCENT / 10), MIN_PRECISION_PERCENT);
  assert.equal(clampPrecisionPercent(MAX_PRECISION_PERCENT * 10), MAX_PRECISION_PERCENT);
  assert.equal(clampPrecisionPercent(Number.NaN), DEFAULT_PRECISION_PERCENT);
});

test("a goal carries the trial limit only when one is set", () => {
  assert.equal(buildStoppingPayload({ ...GOAL_DEFAULTS, maxTrials: 0 }).max_trials, undefined);
  assert.equal(buildStoppingPayload({ ...GOAL_DEFAULTS, maxTrials: 5_000_000 }).max_trials, 5_000_000);
});

test("no time limit is explicit, not an omission", () => {
  const payload = buildStoppingPayload({ ...GOAL_DEFAULTS, timeLimitSeconds: null });
  assert.equal(payload.max_time, "none");
  assert.equal(payload.rel_error, 0.01);
});

test("a fixed design sends trials and no goal", () => {
  const payload = buildStoppingPayload({ ...GOAL_DEFAULTS, mode: "fixed", fixedTrials: 1_000_000 });
  assert.equal(payload.trials, 1_000_000);
  assert.equal(payload.rel_error, undefined);
  assert.equal(payload.abs_error, undefined);
  assert.equal(payload.max_time, 600);
});

test("a fixed design without a time limit omits max_time", () => {
  const payload = buildStoppingPayload({
    ...GOAL_DEFAULTS,
    mode: "fixed",
    fixedTrials: 1_000,
    timeLimitSeconds: null,
  });
  assert.deepEqual(payload, { trials: 1_000, confidence: 0.95 });
});

test("a fixed design needs at least one trial", () => {
  assert.equal(buildStoppingPayload({ ...GOAL_DEFAULTS, mode: "fixed", fixedTrials: 0 }).trials, 1);
});

test("time limits split and rejoin", () => {
  assert.deepEqual(splitTimeLimit(600), { value: 10, unit: "m" });
  assert.deepEqual(splitTimeLimit(7200), { value: 2, unit: "h" });
  assert.deepEqual(splitTimeLimit(90), { value: 90, unit: "s" });
  assert.equal(joinTimeLimit(10, "m"), 600);
  assert.equal(joinTimeLimit(1.5, "h"), 5400);
  assert.equal(formatTimeLimit(600), "10m");
  assert.equal(formatTimeLimit(null), "no limit");
  assert.equal(formatTimeLimit("none"), "no limit");
});

test("the summary names the goal, the level and the limits", () => {
  const summary = describeStopping(
    buildStoppingPayload({ ...GOAL_DEFAULTS, precisionPercent: 2, maxTrials: 10_000_000 }),
  );
  assert.match(summary, /goal ±2% at 95%/);
  assert.match(summary, /max 10,000,000 trials/);
  assert.match(summary, /max 10m/);

  const fixed = describeStopping(buildStoppingPayload({ ...GOAL_DEFAULTS, mode: "fixed", fixedTrials: 500 }));
  assert.match(fixed, /fixed design, 500 trials/);
});

test("the sample-shape budget follows the design", () => {
  assert.equal(trialBudget(GOAL_DEFAULTS), 0);
  assert.equal(trialBudget({ ...GOAL_DEFAULTS, maxTrials: 4096 }), 4096);
  assert.equal(trialBudget({ ...GOAL_DEFAULTS, mode: "fixed", fixedTrials: 512, maxTrials: 4096 }), 512);
});

test("stop reasons have labels and a success notion", () => {
  assert.equal(stopReasonLabel(null), "Running");
  assert.equal(stopReasonLabel("goal"), "Goal met");
  assert.equal(stopReasonLabel("max-trials"), "Trial limit");
  assert.equal(stopReasonLabel("max-time"), "Time limit");
  assert.equal(stopReasonLabel("trials"), "Fixed design complete");
  // Legacy reasons still render.
  assert.equal(stopReasonLabel("converged"), "Converged");
  assert.equal(stopReasonLabel("max_trials"), "Trial limit");
  // Anything unknown is shown rather than hidden.
  assert.equal(stopReasonLabel("weird"), "weird");

  assert.equal(stopReasonIsSuccess("goal"), true);
  assert.equal(stopReasonIsSuccess("trials"), true);
  assert.equal(stopReasonIsSuccess("max-trials"), false);
  assert.equal(stopReasonIsSuccess("max-time"), false);
  assert.equal(stopReasonIsSuccess(null), false);
});
