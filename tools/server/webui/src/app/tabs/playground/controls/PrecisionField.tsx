import { useId, useState } from "react";
import { FieldHintTrigger } from "../../../components/ui/HintTooltip";
import { formatNumber } from "../../../lib/format";
import {
  clampPrecisionPercent,
  formatPrecisionPercent,
  MAX_PRECISION_PERCENT,
  MIN_PRECISION_PERCENT,
  precisionPercentToRelError,
} from "./stoppingSpec";
import { SteppedNumberInputRow } from "./SteppedNumberInputRow";

/** ±% steps offered by the increment/decrement buttons. */
const PRECISION_PRESETS = [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10] as const;
const PRESET_EPSILON = 1e-12;

export const PRECISION_FIELD_HINT = [
  "How precise the answer must be: the half-width of the reported interval, as a percentage of the estimate.",
  "Sampling stops at the end of the first pass at which this is met, unless a limit is reached first.",
  `The default, ${formatPrecisionPercent(1)}, needs about 38,400 hits whatever the probability is.`,
].join(" ");

function stepPrecision(percent: number, direction: 1 | -1): number {
  const current = clampPrecisionPercent(percent);
  if (direction > 0) {
    for (const preset of PRECISION_PRESETS) if (preset > current + PRESET_EPSILON) return preset;
    return PRECISION_PRESETS[PRECISION_PRESETS.length - 1];
  }
  for (let index = PRECISION_PRESETS.length - 1; index >= 0; index -= 1) {
    const preset = PRECISION_PRESETS[index];
    if (preset < current - PRESET_EPSILON) return preset;
  }
  return PRECISION_PRESETS[0];
}

function formatPrecisionInput(percent: number): string {
  return formatPrecisionPercent(percent).replace("±", "").replace("%", "");
}

export function PrecisionField({
  value,
  onChange,
  disabled = false,
}: {
  value: number;
  onChange: (percent: number) => void;
  disabled?: boolean;
}) {
  const fieldId = useId();
  const hintId = `${fieldId}-hint`;
  const errorId = `${fieldId}-error`;
  const [focused, setFocused] = useState(false);
  const [draft, setDraft] = useState(() => formatPrecisionInput(value));
  const [error, setError] = useState<string | null>(null);

  const displayValue = focused ? draft : formatPrecisionInput(value);
  const describedBy = [error ? errorId : null, hintId].filter(Boolean).join(" ") || undefined;

  const commit = (raw: unknown) => {
    const number = typeof raw === "number" ? raw : Number(String(raw).replace("%", "").trim());
    if (!Number.isFinite(number)) {
      setError("Enter a number");
      return false;
    }
    if (number < MIN_PRECISION_PERCENT || number > MAX_PRECISION_PERCENT) {
      setError(`Precision must be between ±${MIN_PRECISION_PERCENT}% and ±${MAX_PRECISION_PERCENT}%`);
      return false;
    }
    setError(null);
    onChange(number);
    setDraft(formatPrecisionInput(number));
    return true;
  };

  const applyStep = (direction: 1 | -1) => {
    const next = stepPrecision(value, direction);
    setError(null);
    onChange(next);
    setDraft(formatPrecisionInput(next));
    setFocused(false);
  };

  return (
    <div className="sticky-panel-group__number-field sticky-panel-group__number-field--precision">
      <div className="sticky-panel-group__number-field-label-row">
        <label className="sticky-panel-group__number-field-label" htmlFor={fieldId}>
          Precision (±%)
        </label>
        <FieldHintTrigger label="Precision help" content={PRECISION_FIELD_HINT} />
      </div>
      <SteppedNumberInputRow
        decrementLabel="Tighten the precision goal"
        incrementLabel="Loosen the precision goal"
        onStep={applyStep}
        disableDecrement={stepPrecision(value, -1) === value}
        disableIncrement={stepPrecision(value, 1) === value}
        disabled={disabled}
        suffix={<span className="sticky-panel-group__number-field-scale" aria-hidden="true">%</span>}
        inputProps={{
          id: fieldId,
          type: "number",
          min: MIN_PRECISION_PERCENT,
          max: MAX_PRECISION_PERCENT,
          step: "any",
          value: displayValue,
          "aria-invalid": error ? true : undefined,
          "aria-describedby": describedBy,
          onFocus: () => {
            setDraft(formatPrecisionInput(value));
            setFocused(true);
          },
          onChange: (event) => {
            const nextText = event.currentTarget.value;
            setDraft(nextText);
            if (error) setError(null);
            if (!focused) commit(nextText);
          },
          onBlur: (event) => {
            const ok = commit(event.currentTarget.value);
            if (!ok) setDraft(formatPrecisionInput(value));
            setFocused(false);
          },
        }}
      />
      <p id={hintId} className="sticky-panel-group__number-field-detail" aria-live="polite">
        {`half-width ≤ ${formatNumber(precisionPercentToRelError(value), 4)} · p̂`}
      </p>
      {error ? (
        <p id={errorId} className="sticky-panel-group__number-field-error" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
