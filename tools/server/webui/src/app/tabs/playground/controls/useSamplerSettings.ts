import { useCallback, useState } from "react";
import {
  DEFAULT_REQUESTED_TRIALS_PER_PASS,
  requestedTrialsPerPassSchema,
} from "./samplerTrialsPerPass";

/**
 * Sampler geometry: the only knobs that are not part of the stopping rule.
 *
 * The stopping rule itself lives in `useRunSettings` (the #224 precision goal,
 * limits and fixed design). `requested_trials_per_pass` stays here because it
 * changes how the device batches work, not when the run stops.
 */
export type SamplerSettings = {
  requestedTrialsPerPass: number;
  setRequestedTrialsPerPass: (trials: number) => void;
};

export function useSamplerSettings(): SamplerSettings {
  const [requestedTrialsPerPass, setRequestedTrialsPerPassState] = useState(DEFAULT_REQUESTED_TRIALS_PER_PASS);

  const setRequestedTrialsPerPass = useCallback((trials: number) => {
    const parsed = requestedTrialsPerPassSchema.safeParse(trials);
    if (parsed.success) setRequestedTrialsPerPassState(parsed.data);
  }, []);

  return { requestedTrialsPerPass, setRequestedTrialsPerPass };
}
