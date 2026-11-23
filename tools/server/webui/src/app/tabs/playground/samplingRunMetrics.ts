import { useSyncExternalStore } from "react";

/** Per-query convergence the run settings display, from the frames the API sends. */
export type SamplingRunMetrics = {
  trialsPerSecond: number | null;
  completedPasses: number | null;
  /** Projected pass count for the goal, from the frames' trial projection. */
  projectedPasses: number | null;
  elapsedSeconds: number | null;
  timeToConvergenceSeconds: number | null;
  /** `rel_half_width` of the reported interval: how close the estimate is now. */
  relHalfWidth: number | null;
  /** The goal's relative half-width, when the run has one. */
  targetRelHalfWidth: number | null;
  /** Why the run stopped: `goal`, `max-trials`, `max-time`, `trials`, … */
  stopReason: string | null;
  /** Whether the goal (not a limit) ended the run. */
  converged: boolean;
};

export type SamplingRunMetricsPatch = Partial<SamplingRunMetrics>;

export type SamplingRunMetricsStore = {
  getSnapshot: () => SamplingRunMetrics;
  subscribe: (listener: () => void) => () => void;
  setMetrics: (metrics: SamplingRunMetricsPatch) => void;
  resetMetrics: () => void;
};

export const EMPTY_SAMPLING_RUN_METRICS: SamplingRunMetrics = {
  trialsPerSecond: null,
  completedPasses: null,
  projectedPasses: null,
  elapsedSeconds: null,
  timeToConvergenceSeconds: null,
  relHalfWidth: null,
  targetRelHalfWidth: null,
  stopReason: null,
  converged: false,
};

function metricsEqual(left: SamplingRunMetrics, right: SamplingRunMetrics): boolean {
  return (
    left.trialsPerSecond === right.trialsPerSecond &&
    left.completedPasses === right.completedPasses &&
    left.projectedPasses === right.projectedPasses &&
    left.elapsedSeconds === right.elapsedSeconds &&
    left.timeToConvergenceSeconds === right.timeToConvergenceSeconds &&
    left.relHalfWidth === right.relHalfWidth &&
    left.targetRelHalfWidth === right.targetRelHalfWidth &&
    left.stopReason === right.stopReason &&
    left.converged === right.converged
  );
}

export function createSamplingRunMetricsStore(
  initialMetrics: SamplingRunMetrics = EMPTY_SAMPLING_RUN_METRICS,
): SamplingRunMetricsStore {
  let snapshot = initialMetrics;
  const listeners = new Set<() => void>();

  return {
    getSnapshot: () => snapshot,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    setMetrics: (metrics) => {
      const next = { ...snapshot, ...metrics };
      if (metricsEqual(snapshot, next)) return;
      snapshot = next;
      listeners.forEach((listener) => listener());
    },
    resetMetrics: () => {
      if (metricsEqual(snapshot, EMPTY_SAMPLING_RUN_METRICS)) return;
      snapshot = EMPTY_SAMPLING_RUN_METRICS;
      listeners.forEach((listener) => listener());
    },
  };
}

export function useSamplingRunMetrics(store: SamplingRunMetricsStore): SamplingRunMetrics {
  return useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);
}
