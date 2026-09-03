import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { RegimeState, FundManagerDecision, DebateResult, IndicatorKey } from "./types";
import { REGIME_NAMES, INDICATOR_DEFS } from "./types";
import type { WsStatus } from "./websocket";

export type AppMode = "historical" | "live";

interface MARTAStore {
  // Mode
  mode: AppMode;
  setMode: (mode: AppMode) => void;

  // Selected asset
  selectedAsset: string;
  setSelectedAsset: (asset: string) => void;

  // Selected timestamp (historical mode)
  selectedTimestamp: string | null;
  setSelectedTimestamp: (ts: string | null) => void;

  // Current regime
  currentRegime: RegimeState | null;
  setCurrentRegime: (regime: RegimeState) => void;

  // Current decision
  currentDecision: FundManagerDecision | null;
  setCurrentDecision: (decision: FundManagerDecision | null) => void;

  // Connection status
  wsConnected: boolean;
  setWsConnected: (connected: boolean) => void;
  wsStatus: WsStatus;
  setWsStatus: (status: WsStatus) => void;
  wsLastUpdate: number | null;
  setWsLastUpdate: (ts: number | null) => void;

  // Active debate
  activeDebate: DebateResult | null;
  setActiveDebate: (debate: DebateResult | null) => void;

  // Demo presentation mode
  demoMode: boolean;
  setDemoMode: (on: boolean) => void;

  // Agent configuration (session-local)
  showContrarian: boolean;
  setShowContrarian: (on: boolean) => void;

  // Approval mode — server-persisted (app_settings.json), not session-local, since the
  // hourly scheduler must see the same value with no browser tab open. This field just
  // mirrors that server state for the UI; Settings page hydrates it on mount and POSTs
  // /api/settings/approval-mode on toggle (see AgentConfigSection / api.setApprovalMode).
  approvalMode: boolean;
  setApprovalMode: (on: boolean) => void;

  // RL execution toggle — server-persisted, same reasoning as approvalMode above.
  // Default off: docs/RESULTS_AND_METHODOLOGY.md Section 9's controlled ablation found
  // no crypto5 RL configuration with a defensible edge over buy-and-hold. LLM consensus
  // + static execution is the default signal source; this is an explicit opt-in.
  rlExecutionEnabled: boolean;
  setRlExecutionEnabled: (on: boolean) => void;

  // Weekly-cadence PPO toggle — a DIFFERENT checkpoint from rlExecutionEnabled above
  // (native decision_interval=168 training, validated with a positive Sharpe on
  // BTC_USDT — see rl/train_weekly_smoketest.py). BTC_USDT-only; takes over that
  // asset's signal entirely when on. Default off, same server-persisted pattern.
  weeklyPpoEnabled: boolean;
  setWeeklyPpoEnabled: (on: boolean) => void;
  weeklyPpoObsMode: "raw" | "raw_regime";
  setWeeklyPpoObsMode: (mode: "raw" | "raw_regime") => void;

  // Sidebar collapse
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;

  // Live regime display names — initialized from the static REGIME_NAMES fallback,
  // overwritten once at app boot from /api/regime/semantics (see AppShell.tsx). The
  // static fallback goes stale every time the HMM is refit, so every REGIME_NAMES[x]
  // usage in the app should read from here instead once this has loaded.
  regimeNames: Record<number, string>;
  setRegimeNames: (names: Record<number, string>) => void;
}

export const useMARTAStore = create<MARTAStore>((set) => ({
  mode: "historical",
  setMode: (mode) => set({ mode }),

  selectedAsset: "BTC_USDT",
  setSelectedAsset: (selectedAsset) => set({ selectedAsset }),

  selectedTimestamp: null,
  setSelectedTimestamp: (selectedTimestamp) => set({ selectedTimestamp }),

  currentRegime: null,
  setCurrentRegime: (currentRegime) => set({ currentRegime }),

  currentDecision: null,
  setCurrentDecision: (currentDecision) => set({ currentDecision }),

  wsConnected: false,
  setWsConnected: (wsConnected) => set({ wsConnected }),
  wsStatus: "idle",
  setWsStatus: (wsStatus) => set({ wsStatus }),
  wsLastUpdate: null,
  setWsLastUpdate: (wsLastUpdate) => set({ wsLastUpdate }),

  activeDebate: null,
  setActiveDebate: (activeDebate) => set({ activeDebate }),

  demoMode: false,
  setDemoMode: (demoMode) => set({ demoMode }),

  showContrarian: true,
  setShowContrarian: (showContrarian) => set({ showContrarian }),

  approvalMode: false,
  setApprovalMode: (approvalMode) => set({ approvalMode }),

  rlExecutionEnabled: false,
  setRlExecutionEnabled: (rlExecutionEnabled) => set({ rlExecutionEnabled }),

  weeklyPpoEnabled: false,
  setWeeklyPpoEnabled: (weeklyPpoEnabled) => set({ weeklyPpoEnabled }),
  weeklyPpoObsMode: "raw_regime",
  setWeeklyPpoObsMode: (weeklyPpoObsMode) => set({ weeklyPpoObsMode }),

  sidebarCollapsed: false,
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),

  regimeNames: { ...REGIME_NAMES },
  setRegimeNames: (regimeNames) => set({ regimeNames }),
}));

// ── Indicator visibility (Settings → Indicators) ──────────────────────────────────
//
// Which of the 13 indicator values show up in the Dashboard's Indicators panel. This is
// a real user preference (not session-local like showContrarian above), so it's a
// separate persisted store rather than a field on useMARTAStore — localStorage-backed,
// survives reloads/tab close, no backend round-trip needed since it's pure display state
// (nothing server-side reads it, unlike approvalMode).

interface IndicatorSettingsStore {
  visible: Record<IndicatorKey, boolean>;
  setVisible: (key: IndicatorKey, visible: boolean) => void;
  showAll: () => void;
  hideAll: () => void;
}

const ALL_INDICATOR_KEYS = INDICATOR_DEFS.map((d) => d.key);

function allIndicators(value: boolean): Record<IndicatorKey, boolean> {
  return Object.fromEntries(ALL_INDICATOR_KEYS.map((k) => [k, value])) as Record<IndicatorKey, boolean>;
}

export const useIndicatorSettingsStore = create<IndicatorSettingsStore>()(
  persist(
    (set) => ({
      // Default: everything visible — Settings lets the user narrow it down.
      visible: allIndicators(true),
      setVisible: (key, value) => set((s) => ({ visible: { ...s.visible, [key]: value } })),
      showAll: () => set({ visible: allIndicators(true) }),
      hideAll: () => set({ visible: allIndicators(false) }),
    }),
    {
      name: "marta-indicator-visibility",
      // Merge stored prefs over current defaults so a newly-added indicator (a future
      // features.py column) defaults to visible instead of `undefined` for users with an
      // older persisted blob.
      merge: (persisted, current) => ({
        ...current,
        ...(persisted as Partial<IndicatorSettingsStore>),
        visible: { ...current.visible, ...(persisted as IndicatorSettingsStore)?.visible },
      }),
    },
  ),
);
