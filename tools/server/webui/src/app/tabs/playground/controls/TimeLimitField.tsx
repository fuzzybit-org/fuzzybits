import { useId, useState } from "react";
import { FieldHintTrigger } from "../../../components/ui/HintTooltip";
import {
  formatTimeLimit,
  joinTimeLimit,
  splitTimeLimit,
  TIME_LIMIT_UNITS,
  type TimeLimitUnit,
} from "./stoppingSpec";
import { SteppedNumberInputRow } from "./SteppedNumberInputRow";

export const TIME_LIMIT_FIELD_HINT = [
  "Wall-clock limit for the run. The run always finishes the pass in progress, so it can overrun by one pass.",
  "Leave it empty (or choose no limit) to run until the goal is met.",
].join(" ");

export function TimeLimitField({
  seconds,
  onChange,
  disabled = false,
}: {
  seconds: number | null;
  onChange: (seconds: number | null) => void;
  disabled?: boolean;
}) {
  const fieldId = useId();
  const unitId = useId();
  const hintId = `${fieldId}-hint`;
  const errorId = `${fieldId}-error`;
  const initial = seconds === null ? { value: 0, unit: "m" as TimeLimitUnit } : splitTimeLimit(seconds);
  const [focused, setFocused] = useState(false);
  const [draft, setDraft] = useState(() => (seconds === null ? "" : String(initial.value)));
  const [unit, setUnit] = useState<TimeLimitUnit>(initial.unit);
  const [error, setError] = useState<string | null>(null);

  const displayValue = focused ? draft : seconds === null ? "" : String(splitTimeLimit(seconds).value);
  const displayedUnit = focused ? unit : seconds === null ? unit : splitTimeLimit(seconds).unit;
  const describedBy = [error ? errorId : null, hintId].filter(Boolean).join(" ") || undefined;

  const commit = (raw: unknown, nextUnit: TimeLimitUnit) => {
    const text = String(raw ?? "").trim();
    if (text === "") {
      setError(null);
      onChange(null);
      setDraft("");
      return true;
    }
    const value = Number(text);
    if (!Number.isFinite(value) || value <= 0) {
      setError("Enter a positive duration, or leave it empty for no limit");
      return false;
    }
    setError(null);
    onChange(joinTimeLimit(value, nextUnit));
    setDraft(String(value));
    return true;
  };

  const applyStep = (direction: 1 | -1) => {
    const current = seconds === null ? { value: 0, unit } : splitTimeLimit(seconds);
    const step = current.value >= 60 ? 60 : 1;
    const value = Math.max(0, current.value + direction * step);
    if (value <= 0) {
      setError(null);
      onChange(null);
      setDraft("");
      setFocused(false);
      return;
    }
    onChange(joinTimeLimit(value, current.unit));
    setDraft(String(value));
    setUnit(current.unit);
    setFocused(false);
  };

  return (
    <div className="sticky-panel-group__number-field sticky-panel-group__number-field--time-limit">
      <div className="sticky-panel-group__number-field-label-row">
        <label className="sticky-panel-group__number-field-label" htmlFor={fieldId}>
          Time limit
        </label>
        <FieldHintTrigger label="Time limit help" content={TIME_LIMIT_FIELD_HINT} />
      </div>
      <SteppedNumberInputRow
        decrementLabel="Decrease the time limit"
        incrementLabel="Increase the time limit"
        onStep={applyStep}
        disabled={disabled}
        suffix={(
          <select
            id={unitId}
            className="sticky-panel-group__number-field-scale sticky-panel-group__number-field-scale-select"
            value={displayedUnit}
            disabled={disabled}
            aria-label="Time limit unit"
            onChange={(event) => {
              const next = event.currentTarget.value as TimeLimitUnit;
              if (!TIME_LIMIT_UNITS.includes(next)) return;
              setUnit(next);
              if (seconds !== null) commit(splitTimeLimit(seconds).value, next);
            }}
          >
            {TIME_LIMIT_UNITS.map((entry) => (
              <option key={entry} value={entry}>
                {entry}
              </option>
            ))}
          </select>
        )}
        inputProps={{
          id: fieldId,
          type: "number",
          min: 0,
          step: "any",
          placeholder: "no limit",
          value: displayValue,
          "aria-invalid": error ? true : undefined,
          "aria-describedby": describedBy,
          onFocus: () => {
            setDraft(seconds === null ? "" : String(splitTimeLimit(seconds).value));
            setUnit(seconds === null ? unit : splitTimeLimit(seconds).unit);
            setFocused(true);
          },
          onChange: (event) => {
            const nextText = event.currentTarget.value;
            setDraft(nextText);
            if (error) setError(null);
            if (!focused) commit(nextText, unit);
          },
          onBlur: (event) => {
            const ok = commit(event.currentTarget.value, unit);
            if (!ok) setDraft(seconds === null ? "" : String(splitTimeLimit(seconds).value));
            setFocused(false);
          },
        }}
      />
      <p id={hintId} className="sticky-panel-group__number-field-detail" aria-live="polite">
        {seconds === null ? "no limit" : formatTimeLimit(seconds)}
      </p>
      {error ? (
        <p id={errorId} className="sticky-panel-group__number-field-error" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
