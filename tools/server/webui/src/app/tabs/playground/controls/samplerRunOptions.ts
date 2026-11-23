import type { StoppingPayload } from "./stoppingSpec";

/**
 * Run-settings body for `POST /api/v1/start` and `/api/v1/reset`.
 *
 * The stopping rule goes in `stopping` (the #224 specification). The legacy flat
 * options (`num_trials`, `early_stop`, `delta`, `burn_in`, `policy`) are still
 * accepted by the server for one release, but this client does not send them:
 * there is one vocabulary, and it is the specification.
 */
export type SamplerRunOptions = {
  model_id?: string;
  compilation_level?: number;
  expand_atleast_gates?: boolean;
  expand_xor_gates?: boolean;
  recognize_atleast_groups?: boolean;
  recognize_xor_groups?: boolean;
  ccf?: boolean;
  keep_null_gates?: boolean;
  /** Precision goal, limits, or a fixed trial count (#224). */
  stopping: StoppingPayload;
  /** Optimistic per-node trials-per-pass target (`settings.requested_trials_per_pass`; 0 = auto). */
  requested_trials_per_pass: number;
  /** Known true probability µ (mcSCRAM `oracle`); omitted when diagnostics are off. */
  oracle?: number;
};
