import { StickyPanelGroup, type StickyPanelSpan } from "../../../components/ui/StickyPanel";
import { ConfidenceCiField } from "./ConfidenceCiField";
import { FixedTrialsField } from "./FixedTrialsField";
import { PrecisionField } from "./PrecisionField";
import { TimeLimitField } from "./TimeLimitField";
import { TrialsNField } from "./TrialsNField";
import type { RunSettings } from "./useRunSettings";

const MAX_TRIALS_FIELD_HINT =
  "The most trials the run may use. Leave it empty for no trial limit; the time limit and the precision goal then decide when it stops.";

const FIXED_TRIALS_FIELD_HINT =
  "Exactly how many trials the run will execute. A fixed design has no early stop, so the precision goal does not apply.";

/**
 * Run settings: the #224 stopping specification as three separate answers —
 * the goal (precision, confidence), the limits (trials, time) and the design
 * (fixed or not). A fixed design disables the goal fields, because `-N` and a
 * goal cannot be combined.
 */
export function RunSettingsControlGroup({
  settings,
  disabled = false,
  hSpan,
  vSpan,
}: {
  settings: RunSettings;
  disabled?: boolean;
  hSpan: StickyPanelSpan;
  vSpan: StickyPanelSpan;
}) {
  const fixed = settings.mode === "fixed";
  const goalDisabled = disabled || fixed;

  return (
    <StickyPanelGroup legend="Stopping" className="sticky-panel-group--run-settings" hSpan={hSpan} vSpan={vSpan}>
      <div className="pg-fields">
        <FixedTrialsField mode={settings.mode} onChange={settings.setMode} disabled={disabled} />
        {fixed ? (
          <TrialsNField
            mantissa={settings.fixedTrialsMantissa}
            scaleExponent={settings.fixedTrialsScaleExponent}
            onMantissaChange={settings.setFixedTrialsMantissa}
            onScaleExponentChange={settings.setFixedTrialsScaleExponent}
            label="Fixed trials"
            allowEmpty={false}
            hint={FIXED_TRIALS_FIELD_HINT}
            disabled={disabled}
          />
        ) : (
          <PrecisionField
            value={settings.precisionPercent}
            onChange={settings.setPrecisionPercent}
            disabled={goalDisabled}
          />
        )}
        <ConfidenceCiField
          value={settings.confidencePercent}
          onChange={settings.setConfidencePercent}
          disabled={goalDisabled}
        />
        <TrialsNField
          mantissa={settings.maxTrialsMantissa}
          scaleExponent={settings.maxTrialsScaleExponent}
          onMantissaChange={settings.setMaxTrialsMantissa}
          onScaleExponentChange={settings.setMaxTrialsScaleExponent}
          label={fixed ? "Trial limit" : "Max trials"}
          hint={MAX_TRIALS_FIELD_HINT}
          allowEmpty
          disabled={disabled}
        />
        <TimeLimitField
          seconds={settings.timeLimitSeconds}
          onChange={settings.setTimeLimitSeconds}
          disabled={disabled}
        />
        <p className="sticky-panel-group__number-field-detail" aria-live="polite">
          {settings.summary}
        </p>
      </div>
    </StickyPanelGroup>
  );
}
