import { FieldHintTrigger } from "../../../components/ui/HintTooltip";
import { StickyPanelCheckbox } from "../../../components/ui/StickyPanel";
import type { StoppingMode } from "./stoppingSpec";

export const FIXED_TRIALS_FIELD_HINT =
  "On: run exactly the number of trials in Max Trials, with no early stop. Off: stop when the precision goal is met, or at a limit. A fixed design has no goal, so the precision and confidence fields are disabled.";

export function FixedTrialsField({
  mode,
  onChange,
  disabled = false,
}: {
  mode: StoppingMode;
  onChange: (mode: StoppingMode) => void;
  disabled?: boolean;
}) {
  return (
    <div className="sticky-panel-group__toggle-row">
      <StickyPanelCheckbox
        id="playground-fixed-trials"
        label="FIXED NUMBER OF TRIALS"
        checked={mode === "fixed"}
        onChange={(checked) => onChange(checked ? "fixed" : "goal")}
        disabled={disabled}
      />
      <FieldHintTrigger label="Fixed number of trials help" content={FIXED_TRIALS_FIELD_HINT} />
    </div>
  );
}
