import { useCallback, useMemo, useState } from "react";
import {
  createTrialsMantissaSchema,
  DEFAULT_TRIALS_MANTISSA,
  DEFAULT_TRIALS_SCALE_EXPONENT,
  effectiveTrialsCount,
  isTrialsScaleExponent,
  normalizeTrialsMantissa,
  type TrialsScaleExponent,
} from "./samplerTrials";
import {
  buildStoppingPayload,
  clampPrecisionPercent,
  DEFAULT_CONFIDENCE_PERCENT,
  DEFAULT_PRECISION_PERCENT,
  DEFAULT_TIME_LIMIT_MINUTES,
  describeStopping,
  precisionPercentToRelError,
  type RunSettingsInput,
  type StoppingMode,
  type StoppingPayload,
} from "./stoppingSpec";

/**
 * The run-settings form: a precision goal or a fixed design, plus the limits.
 *
 * Trial counts use the mantissa/scale pair the sampler field already uses, so a
 * budget reads as "1 B" rather than nine digits. `0` (an empty field) means "no
 * limit" for the trial limit, and is not allowed for a fixed design.
 */
export type RunSettings = {
  mode: StoppingMode;
  setMode: (mode: StoppingMode) => void;
  precisionPercent: number;
  setPrecisionPercent: (percent: number) => void;
  confidencePercent: number;
  setConfidencePercent: (percent: number) => void;
  /** Wall-clock limit in seconds; null means none. */
  timeLimitSeconds: number | null;
  setTimeLimitSeconds: (seconds: number | null) => void;
  /** Trial limit mantissa/scale; a derived count of 0 means no limit. */
  maxTrialsMantissa: number;
  maxTrialsScaleExponent: TrialsScaleExponent;
  setMaxTrialsMantissa: (mantissa: number) => void;
  setMaxTrialsScaleExponent: (exponent: TrialsScaleExponent) => void;
  /** Derived trial limit; 0 means none. */
  maxTrials: number;
  /** Fixed-design mantissa/scale. */
  fixedTrialsMantissa: number;
  fixedTrialsScaleExponent: TrialsScaleExponent;
  setFixedTrialsMantissa: (mantissa: number) => void;
  setFixedTrialsScaleExponent: (exponent: TrialsScaleExponent) => void;
  /** Derived fixed trial count. */
  fixedTrials: number;
  /** The `stopping` object sent to the server. */
  stopping: StoppingPayload;
  /** One-line description of the run. */
  summary: string;
  /** The goal as a relative half-width. */
  relError: number;
};

export const DEFAULT_RUN_SETTINGS = {
  mode: "goal" as StoppingMode,
  precisionPercent: DEFAULT_PRECISION_PERCENT,
  confidencePercent: DEFAULT_CONFIDENCE_PERCENT,
  timeLimitSeconds: DEFAULT_TIME_LIMIT_MINUTES * 60,
  maxTrialsMantissa: DEFAULT_TRIALS_MANTISSA,
  maxTrialsScaleExponent: DEFAULT_TRIALS_SCALE_EXPONENT,
  fixedTrialsMantissa: 1,
  fixedTrialsScaleExponent: 9 as TrialsScaleExponent,
};

export function useRunSettings(): RunSettings {
  const [mode, setMode] = useState<StoppingMode>(DEFAULT_RUN_SETTINGS.mode);
  const [precisionPercent, setPrecisionPercentState] = useState(DEFAULT_RUN_SETTINGS.precisionPercent);
  const [confidencePercent, setConfidencePercentState] = useState(DEFAULT_RUN_SETTINGS.confidencePercent);
  const [timeLimitSeconds, setTimeLimitSecondsState] = useState<number | null>(
    DEFAULT_RUN_SETTINGS.timeLimitSeconds,
  );
  const [maxTrialsMantissa, setMaxTrialsMantissaState] = useState(DEFAULT_RUN_SETTINGS.maxTrialsMantissa);
  const [maxTrialsScaleExponent, setMaxTrialsScaleExponentState] = useState<TrialsScaleExponent>(
    DEFAULT_RUN_SETTINGS.maxTrialsScaleExponent,
  );
  const [fixedTrialsMantissa, setFixedTrialsMantissaState] = useState(DEFAULT_RUN_SETTINGS.fixedTrialsMantissa);
  const [fixedTrialsScaleExponent, setFixedTrialsScaleExponentState] = useState<TrialsScaleExponent>(
    DEFAULT_RUN_SETTINGS.fixedTrialsScaleExponent,
  );

  const setPrecisionPercent = useCallback((percent: number) => {
    setPrecisionPercentState(clampPrecisionPercent(percent));
  }, []);
  const setConfidencePercent = useCallback((percent: number) => {
    if (!Number.isFinite(percent)) return;
    setConfidencePercentState(Math.min(99.999, Math.max(50, Number(percent.toFixed(4)))));
  }, []);
  const setTimeLimitSeconds = useCallback((seconds: number | null) => {
    setTimeLimitSecondsState(seconds !== null && seconds > 0 ? seconds : null);
  }, []);
  const setMaxTrialsMantissa = useCallback(
    (mantissa: number) => {
      if (mantissa === 0) {
        setMaxTrialsMantissaState(0);
        return;
      }
      const parsed = createTrialsMantissaSchema(maxTrialsScaleExponent).safeParse(mantissa);
      if (parsed.success) setMaxTrialsMantissaState(normalizeTrialsMantissa(parsed.data));
    },
    [maxTrialsScaleExponent],
  );
  const setMaxTrialsScaleExponent = useCallback((exponent: TrialsScaleExponent) => {
    if (isTrialsScaleExponent(exponent)) setMaxTrialsScaleExponentState(exponent);
  }, []);
  const setFixedTrialsMantissa = useCallback(
    (mantissa: number) => {
      const parsed = createTrialsMantissaSchema(fixedTrialsScaleExponent).safeParse(mantissa);
      // A fixed design needs at least one trial; a zero mantissa is not a design.
      if (parsed.success && parsed.data > 0) setFixedTrialsMantissaState(normalizeTrialsMantissa(parsed.data));
    },
    [fixedTrialsScaleExponent],
  );
  const setFixedTrialsScaleExponent = useCallback((exponent: TrialsScaleExponent) => {
    if (isTrialsScaleExponent(exponent)) setFixedTrialsScaleExponentState(exponent);
  }, []);

  const maxTrials = useMemo(
    () => effectiveTrialsCount(maxTrialsMantissa, maxTrialsScaleExponent),
    [maxTrialsMantissa, maxTrialsScaleExponent],
  );
  const fixedTrials = useMemo(
    () => effectiveTrialsCount(fixedTrialsMantissa, fixedTrialsScaleExponent),
    [fixedTrialsMantissa, fixedTrialsScaleExponent],
  );

  const input: RunSettingsInput = {
    mode,
    precisionPercent,
    confidencePercent,
    timeLimitSeconds,
    maxTrials,
    fixedTrials,
  };
  const stopping = useMemo(
    () => buildStoppingPayload(input),
    [mode, precisionPercent, confidencePercent, timeLimitSeconds, maxTrials, fixedTrials],
  );
  const summary = useMemo(() => describeStopping(stopping), [stopping]);

  return {
    mode,
    setMode,
    precisionPercent,
    setPrecisionPercent,
    confidencePercent,
    setConfidencePercent,
    timeLimitSeconds,
    setTimeLimitSeconds,
    maxTrialsMantissa,
    maxTrialsScaleExponent,
    setMaxTrialsMantissa,
    setMaxTrialsScaleExponent,
    maxTrials,
    fixedTrialsMantissa,
    fixedTrialsScaleExponent,
    setFixedTrialsMantissa,
    setFixedTrialsScaleExponent,
    fixedTrials,
    stopping,
    summary,
    relError: precisionPercentToRelError(precisionPercent),
  };
}
