import { StickyPanelGroup, type StickyPanelSpan } from "../../../components/ui/StickyPanel";
import { TrialsPerPassField } from "./TrialsPerPassField";
import type { SamplerSettings } from "./useSamplerSettings";

/** Device geometry only: the stopping rule is the `Stopping` group. */
export function SamplerControlGroup({
  settings,
  disabled = false,
  hSpan,
  vSpan,
}: {
  settings: SamplerSettings;
  disabled?: boolean;
  hSpan: StickyPanelSpan;
  vSpan: StickyPanelSpan;
}) {
  return (
    <StickyPanelGroup legend="Sampler" className="sticky-panel-group--sampler" hSpan={hSpan} vSpan={vSpan}>
      <div className="pg-fields">
        <TrialsPerPassField
          value={settings.requestedTrialsPerPass}
          onChange={settings.setRequestedTrialsPerPass}
          disabled={disabled}
        />
      </div>
    </StickyPanelGroup>
  );
}
