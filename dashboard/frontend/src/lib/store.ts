import { create } from "zustand";
import type { RegimeState, FundManagerDecision, DebateResult } from "./types";
import { REGIME_NAMES } from "./types";
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

  sidebarCollapsed: false,
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),

  regimeNames: { ...REGIME_NAMES },
  setRegimeNames: (regimeNames) => set({ regimeNames }),
}));
