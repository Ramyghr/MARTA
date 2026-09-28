"use client";
import { useEffect, useState } from "react";
import { ExternalLink } from "lucide-react";
import { Panel } from "@/components/ui/Panel";
import { PageHeader } from "@/components/layout/PageHeader";
import { Badge } from "@/components/ui/Badge";
import { Spinner } from "@/components/ui/Spinner";
import { SkeletonRows } from "@/components/ui/Skeleton";
import { LearningPanel } from "@/components/LearningPanel";
import Link from "next/link";
import { User } from "lucide-react";
import { fetchApi, API_URL, WS_URL } from "@/lib/websocket";
import { useMARTAStore, useIndicatorSettingsStore } from "@/lib/store";
import { api } from "@/lib/api";
import { useCurrentUser } from "@/lib/auth";
import { INDICATOR_DEFS } from "@/lib/types";
import type { SystemStatus, SystemModels } from "@/lib/api";

interface AssetFreshness {
  id: string;
  label: string;
  type: string;
  last_timestamp: string | null;
  age_hours: number;
  stale: boolean;
}
interface Freshness {
  assets: AssetFreshness[];
  any_stale: boolean;
  cron_last_ran: string | null;
  checked_at: string;
  stale_threshold_hours: number;
}

function ConfigRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-[#161616] last:border-0 text-[11px]">
      <span className="text-[#555]">{label}</span>
      <span className="text-[#999] font-mono">{value}</span>
    </div>
  );
}

// ── Section 1 — Data Sources ───────────────────────────────────────────────────

interface SourceState { tested: boolean; ok: boolean; message: string; testing: boolean }
const EMPTY_SOURCE_STATE: SourceState = { tested: false, ok: false, message: "", testing: false };

function DataSourceRow({ id, label, keyLabel, secretLabel, subLabel, customTest }: {
  id: string;
  label: string;
  keyLabel?: string;
  /** Adds a second password input (e.g. Alpaca needs an API key + a secret). */
  secretLabel?: string;
  /** Shown under the label, e.g. "No API key required — public stream" */
  subLabel?: string;
  /** Overrides the default api.testSource(id) call — used for sources with their
   * own dedicated endpoint (StockTwits, Fear & Greed) instead of the generic
   * /api/system/test-source catch-all. */
  customTest?: () => Promise<{ ok: boolean; message: string }>;
}) {
  const [state, setState] = useState<SourceState>(EMPTY_SOURCE_STATE);
  const [key, setKey] = useState("");
  const [secret, setSecret] = useState("");

  const test = async () => {
    setState((s) => ({ ...s, testing: true }));
    try {
      const r = customTest ? await customTest() : await api.testSource(id);
      setState({ tested: true, ok: r.ok, message: r.message, testing: false });
    } catch {
      setState({ tested: true, ok: false, message: "Request failed", testing: false });
    }
  };

  const dotColor = !state.tested ? "#444" : state.ok ? "#00ff7f" : "#ff3737";

  return (
    <div className="flex flex-col gap-1.5 py-2.5 border-b border-[#161616] last:border-0">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: dotColor, boxShadow: state.tested && state.ok ? `0 0 5px ${dotColor}` : "none" }} />
          <span className="text-[11px] text-[#999] truncate">{label}</span>
        </div>
        <button
          onClick={test}
          disabled={state.testing}
          className="text-[10px] px-2 py-1 rounded border border-[#2a2a2a] text-[#666] hover:text-[#00ff7f] hover:border-[#00ff7f44] transition-colors disabled:opacity-40 shrink-0"
        >
          {state.testing ? "Testing…" : "Test Connection"}
        </button>
      </div>
      {subLabel && <div className="text-[10px] text-[#3a3a3a]">{subLabel}</div>}
      {keyLabel && (
        <input
          type="password"
          value={key}
          onChange={(e) => setKey(e.target.value)}
          placeholder={`${keyLabel} API key (session only, not persisted)`}
          className="bg-[#111] border border-[#1a1a1a] rounded px-2 py-1 text-[10px] text-[#888] outline-none"
        />
      )}
      {secretLabel && (
        <input
          type="password"
          value={secret}
          onChange={(e) => setSecret(e.target.value)}
          placeholder={`${secretLabel} (session only, not persisted)`}
          className="bg-[#111] border border-[#1a1a1a] rounded px-2 py-1 text-[10px] text-[#888] outline-none"
        />
      )}
      {state.tested && (
        <div className="text-[10px]" style={{ color: state.ok ? "#00ff7f" : "#666" }}>{state.message}</div>
      )}
    </div>
  );
}

function DataSourcesSection() {
  return (
    <div className="flex flex-col">
      <DataSourceRow id="binance" label="Binance WebSocket (crypto prices)" />
      <DataSourceRow id="rss" label="RSS — CoinDesk / CoinTelegraph / Decrypt / Reuters" />
      <DataSourceRow id="twelve_data" label="Twelve Data (EUR/USD, XAU/USD)" keyLabel="Twelve Data" />
      <DataSourceRow
        id="telegram"
        label="Telegram Bot (@marta_trading_bot)"
        subLabel="Sends regime shifts and trade signals to your Telegram"
      />
      <DataSourceRow
        id="stocktwits"
        label="StockTwits (financial social sentiment)"
        subLabel="No API key required — public stream"
        customTest={async () => {
          const r = await api.getStockTwits("BTC_USDT");
          return { ok: r.count > 0, message: `StockTwits — ${r.count} messages fetched for BTC` };
        }}
      />
      <DataSourceRow
        id="fear_greed"
        label="Fear & Greed Index (alternative.me)"
        subLabel="No API key required — updates daily"
        customTest={async () => {
          const r = await api.getFearGreed();
          return { ok: true, message: `Fear & Greed: ${r.value} — ${r.classification}` };
        }}
      />
      <DataSourceRow id="newsapi" label="NewsAPI" keyLabel="NewsAPI" />
      <DataSourceRow
        id="alpaca"
        label="Alpaca Paper Trading"
        keyLabel="Alpaca API Key"
        secretLabel="Alpaca Secret Key"
        subLabel="Free practice account at alpaca.markets — set ALPACA_API_KEY / ALPACA_SECRET_KEY in .env to enable (takes priority over Binance testnet for BTC/ETH order execution)"
        customTest={async () => {
          const r = await api.getAlpacaAccount();
          if (!r.configured) return { ok: false, message: "Enter API keys (.env) to enable Alpaca paper trading" };
          if (r.error) return { ok: false, message: `Alpaca error: ${r.error}` };
          return { ok: true, message: `Connected — portfolio value $${(r.portfolio_value ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}` };
        }}
      />
    </div>
  );
}

// ── Section 2 — Model Configuration ────────────────────────────────────────────

function ModelConfigSection() {
  const [models, setModels] = useState<SystemModels | null>(null);

  useEffect(() => {
    api.getSystemModels().then(setModels).catch(() => setModels(null));
  }, []);

  if (!models) return <div className="p-1"><SkeletonRows rows={8} /></div>;

  return (
    <div className="flex flex-col gap-3 text-[11px]">
      <div>
        <div className="stat-label mb-1">Encoder — TFT + InfoNCE</div>
        <ConfigRow label="Checkpoint" value={models.encoder.checkpoint_exists ? "best_encoder.pt" : "missing"} />
        <ConfigRow label="Best Epoch" value={models.encoder.epoch != null ? String(models.encoder.epoch) : "—"} />
        <ConfigRow label="Val Loss" value={models.encoder.val_loss != null ? models.encoder.val_loss.toFixed(4) : "—"} />
        <ConfigRow label="Val Top-1 Accuracy" value={models.encoder.val_accuracy != null ? `${(models.encoder.val_accuracy * 100).toFixed(1)}%` : "Check MLflow"} />
      </div>
      <div>
        <div className="stat-label mb-1">HMM Regime Model</div>
        <ConfigRow label="n_states" value={models.hmm.n_states != null ? String(models.hmm.n_states) : "—"} />
        <ConfigRow label="Semantic Validation" value={models.hmm.semantic_validation_passed == null ? "—" : models.hmm.semantic_validation_passed ? "PASSED" : "FAILED"} />
      </div>
      <div>
        <div className="stat-label mb-1">RL Execution — PPO</div>
        <ConfigRow label="Checkpoint" value={models.rl.checkpoint_exists ? "ppo_best.zip" : "missing"} />
        <ConfigRow label="Best Sharpe" value={models.rl.best_sharpe != null ? models.rl.best_sharpe.toFixed(3) : "—"} />
        <ConfigRow label="Checkpoint Date" value={models.rl.checkpoint_mtime ? new Date(models.rl.checkpoint_mtime).toLocaleDateString() : "—"} />
      </div>
      <a
        href="http://localhost:5000"
        target="_blank"
        rel="noopener noreferrer"
        className="mt-1 flex items-center justify-center gap-1.5 py-1.5 rounded text-[10px] font-semibold uppercase tracking-wider bg-[#4a9eff]/10 text-[#4a9eff] border border-[#4a9eff]/30 hover:bg-[#4a9eff]/20 transition-colors"
      >
        <ExternalLink size={11} /> View MLflow
      </a>
    </div>
  );
}

// ── Section 3 — Agent Configuration ────────────────────────────────────────────

function AgentConfigSection() {
  const showContrarian = useMARTAStore((s) => s.showContrarian);
  const setShowContrarian = useMARTAStore((s) => s.setShowContrarian);
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [model, setModel] = useState("mistral:7b-instruct");
  const [temperature, setTemperature] = useState(0.1);
  const [timeoutSec, setTimeoutSec] = useState(300);

  useEffect(() => {
    api.getSystemStatus().then((s) => {
      const names = s.ollama.models ?? [];
      setOllamaModels(names);
      if (names.length && !names.includes(model)) setModel(names[0]);
    }).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex flex-col gap-4 text-[11px]">
      <div className="text-[10px] text-[#ff8c00] bg-[#ff8c0012] border border-[#ff8c0033] rounded px-2 py-1.5 leading-relaxed">
        Model, temperature, and timeout below are <strong>not wired to the live debate backend</strong> —
        shown disabled so they don&apos;t look like live controls. <code className="text-[#ccc]">agents/base.py</code>{" "}
        hardcodes temperature=0.1 regardless of what&apos;s picked here. &ldquo;Show Contrarian&rdquo; at the
        bottom is the one control on this panel that&apos;s actually live.
      </div>

      <div className="opacity-50">
        <label className="stat-label block mb-1.5">LLM Model</label>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          disabled
          className="w-full bg-[#111] border border-[#2a2a2a] rounded px-2 py-1.5 text-[11px] text-[#ccc] cursor-not-allowed"
        >
          {(ollamaModels.length ? ollamaModels : [model]).map((m) => <option key={m} value={m}>{m}</option>)}
        </select>
        <div className="text-[10px] text-[#444] mt-1">Live from Ollama {ollamaModels.length ? `(${ollamaModels.length} model${ollamaModels.length > 1 ? "s" : ""} pulled)` : "— unreachable"}</div>
      </div>

      <div className="opacity-50">
        <div className="flex items-center justify-between mb-1.5">
          <label className="stat-label">Temperature</label>
          <span className="text-[#999] font-mono">{temperature.toFixed(1)}</span>
        </div>
        <input
          type="range"
          min={0.1}
          max={0.9}
          step={0.1}
          value={temperature}
          onChange={(e) => setTemperature(parseFloat(e.target.value))}
          disabled
          className="w-full cursor-not-allowed"
        />
      </div>

      <div className="opacity-50">
        <label className="stat-label block mb-1.5">Debate Timeout (seconds)</label>
        <input
          type="number"
          min={30}
          max={600}
          step={30}
          value={timeoutSec}
          onChange={(e) => setTimeoutSec(parseInt(e.target.value, 10) || 300)}
          disabled
          className="w-full bg-[#111] border border-[#2a2a2a] rounded px-2 py-1.5 text-[11px] text-[#ccc] cursor-not-allowed"
        />
      </div>

      <label className="flex items-center justify-between cursor-pointer">
        <span className="stat-label">Show Contrarian Agent</span>
        <button
          onClick={() => setShowContrarian(!showContrarian)}
          className="relative w-9 h-5 rounded-full transition-colors shrink-0"
          style={{ background: showContrarian ? "#00ff7f33" : "#1a1a1a" }}
        >
          <span
            className="absolute top-0.5 w-4 h-4 rounded-full transition-all"
            style={{ left: showContrarian ? 18 : 2, background: showContrarian ? "#00ff7f" : "#555" }}
          />
        </button>
      </label>
    </div>
  );
}

// ── Section 3b — Execution Controls ────────────────────────────────────────────

function ExecutionControlsSection() {
  const approvalMode = useMARTAStore((s) => s.approvalMode);
  const setApprovalMode = useMARTAStore((s) => s.setApprovalMode);
  const [saving, setSaving] = useState(false);

  const toggle = async () => {
    const next = !approvalMode;
    setSaving(true);
    setApprovalMode(next); // optimistic — this is a single boolean flip, worth the snappy UI
    try {
      const res = await api.setApprovalMode(next);
      setApprovalMode(res.approval_mode);
    } catch {
      setApprovalMode(!next); // revert on failure
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex flex-col gap-3 text-[11px]">
      <label className="flex items-center justify-between cursor-pointer">
        <span className="stat-label">Require Approval Before Execution</span>
        <button
          onClick={toggle}
          disabled={saving}
          className="relative w-9 h-5 rounded-full transition-colors shrink-0 disabled:opacity-50"
          style={{ background: approvalMode ? "#ff8c0033" : "#1a1a1a" }}
        >
          <span
            className="absolute top-0.5 w-4 h-4 rounded-full transition-all"
            style={{ left: approvalMode ? 18 : 2, background: approvalMode ? "#ff8c00" : "#555" }}
          />
        </button>
      </label>
      <div className="text-[10px] text-[#3a3a3a] leading-relaxed border-t border-[#161616] pt-2">
        {approvalMode ? (
          <>
            <span style={{ color: "#ff8c00" }}>ON</span> — the hourly scheduler and manual
            &ldquo;Run Debate &amp; Execute&rdquo;/&ldquo;Trigger Manual Signal&rdquo; park any non-hold
            signal in a pending queue instead of trading immediately. Review and approve or reject
            each one on the Trading page. Holds never need approval — nothing is at risk from them.
          </>
        ) : (
          "OFF — signals execute immediately against the paper ledger, as before."
        )}
      </div>
    </div>
  );
}

// ── Section 3b-2 — RL Execution (experimental, opt-in) ─────────────────────────

function RlExecutionSection() {
  const rlExecutionEnabled = useMARTAStore((s) => s.rlExecutionEnabled);
  const setRlExecutionEnabled = useMARTAStore((s) => s.setRlExecutionEnabled);
  const [saving, setSaving] = useState(false);

  const toggle = async () => {
    const next = !rlExecutionEnabled;
    setSaving(true);
    setRlExecutionEnabled(next);
    try {
      const res = await api.setRlExecution(next);
      setRlExecutionEnabled(res.rl_execution_enabled);
    } catch {
      setRlExecutionEnabled(!next);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex flex-col gap-3 text-[11px]">
      <label className="flex items-center justify-between cursor-pointer">
        <span className="stat-label">RL Execution (experimental)</span>
        <button
          onClick={toggle}
          disabled={saving}
          className="relative w-9 h-5 rounded-full transition-colors shrink-0 disabled:opacity-50"
          style={{ background: rlExecutionEnabled ? "#ff373733" : "#1a1a1a" }}
        >
          <span
            className="absolute top-0.5 w-4 h-4 rounded-full transition-all"
            style={{ left: rlExecutionEnabled ? 18 : 2, background: rlExecutionEnabled ? "#ff3737" : "#555" }}
          />
        </button>
      </label>
      {rlExecutionEnabled && (
        <div
          className="text-[10px] leading-relaxed rounded px-2.5 py-2 border"
          style={{ background: "#ff373714", borderColor: "#ff373744", color: "#ff8f8f" }}
        >
          <strong>⚠ Warning:</strong> a complete, controlled ablation (regime conditioning,
          reward formulation, behavior-cloning warm-start, discrete vs. continuous action
          space) found no crypto5 PPO configuration with a statistically defensible edge
          over buy-and-hold. Enabling this trades on a model without a confirmed advantage —
          see docs/RESULTS_AND_METHODOLOGY.md Section 9 for the full results.
        </div>
      )}
      <div className="text-[10px] text-[#3a3a3a] leading-relaxed border-t border-[#161616] pt-2">
        {rlExecutionEnabled
          ? "ON — the trained PPO checkpoint drives the signal when available, falling back to the rule below if it isn't."
          : "OFF (default) — signals come from LLM multi-agent consensus, falling back to a static regime-based rule when no recent debate exists."}
      </div>
    </div>
  );
}

// ── Section 3b-3 — Weekly PPO (experimental, opt-in, BTC/USDT only) ────────────

function WeeklyPpoSection() {
  const weeklyPpoEnabled = useMARTAStore((s) => s.weeklyPpoEnabled);
  const setWeeklyPpoEnabled = useMARTAStore((s) => s.setWeeklyPpoEnabled);
  const weeklyPpoObsMode = useMARTAStore((s) => s.weeklyPpoObsMode);
  const setWeeklyPpoObsMode = useMARTAStore((s) => s.setWeeklyPpoObsMode);
  const [saving, setSaving] = useState(false);

  const toggle = async () => {
    const next = !weeklyPpoEnabled;
    setSaving(true);
    setWeeklyPpoEnabled(next);
    try {
      const res = await api.setWeeklyPpo(next);
      setWeeklyPpoEnabled(res.weekly_ppo_enabled);
      setWeeklyPpoObsMode(res.weekly_ppo_obs_mode);
    } catch {
      setWeeklyPpoEnabled(!next);
    } finally {
      setSaving(false);
    }
  };

  const changeObsMode = async (mode: "raw" | "raw_regime") => {
    setSaving(true);
    setWeeklyPpoObsMode(mode);
    try {
      const res = await api.setWeeklyPpo(weeklyPpoEnabled, mode);
      setWeeklyPpoObsMode(res.weekly_ppo_obs_mode);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex flex-col gap-3 text-[11px]">
      <label className="flex items-center justify-between cursor-pointer">
        <span className="stat-label">Weekly PPO — BTC/USDT only (experimental)</span>
        <button
          onClick={toggle}
          disabled={saving}
          className="relative w-9 h-5 rounded-full transition-colors shrink-0 disabled:opacity-50"
          style={{ background: weeklyPpoEnabled ? "#00ff7f33" : "#1a1a1a" }}
        >
          <span
            className="absolute top-0.5 w-4 h-4 rounded-full transition-all"
            style={{ left: weeklyPpoEnabled ? 18 : 2, background: weeklyPpoEnabled ? "#00ff7f" : "#555" }}
          />
        </button>
      </label>
      <div className="flex items-center justify-between">
        <span className="text-[#555]">Observation mode</span>
        <div className="flex rounded overflow-hidden border border-[#1a1a1a]">
          {(["raw", "raw_regime"] as const).map((m) => (
            <button
              key={m}
              onClick={() => changeObsMode(m)}
              disabled={saving}
              className="px-2 py-0.5 text-[10px] transition-colors disabled:opacity-50"
              style={{
                background: weeklyPpoObsMode === m ? "rgba(0,255,127,0.15)" : "transparent",
                color: weeklyPpoObsMode === m ? "#00ff7f" : "#444",
                borderRight: m !== "raw_regime" ? "1px solid #1a1a1a" : "none",
              }}
            >
              {m === "raw" ? "Raw" : "Regime-aware"}
            </button>
          ))}
        </div>
      </div>
      <div className="text-[10px] text-[#3a3a3a] leading-relaxed border-t border-[#161616] pt-2">
        {weeklyPpoEnabled
          ? `ON — replaces the hourly PPO/regime-policy/debate-override chain for BTC/USDT with a policy trained to decide once a week (168h). Validated positive Sharpe on both modes; unaffected by other assets or the RL Execution toggle above.`
          : "OFF (default) — BTC/USDT uses the same signal chain as every other asset."}
      </div>
      <div className="text-[10px] text-[#3a3a3a] leading-relaxed">
        BTC/USDT-only everywhere this policy is used — live execution here and the{" "}
        <code className="text-[#555]">weekly_ppo_raw</code>/<code className="text-[#555]">weekly_ppo_raw_regime</code>{" "}
        strategies on the Strategy Tester (Backtest page) both hard-gate to BTC/USDT. Seed-0 pilot
        checkpoints for ETH/BNB/LTC/XRP already exist on disk from the crypto5 weekly grid, but
        neither live execution nor the Strategy Tester loads them yet — extending to the other four
        assets is a separate change, not done here.
      </div>
    </div>
  );
}

// ── Section 3c — Indicator Visibility ──────────────────────────────────────────

function IndicatorVisibilitySection() {
  const visible = useIndicatorSettingsStore((s) => s.visible);
  const setVisible = useIndicatorSettingsStore((s) => s.setVisible);
  const showAll = useIndicatorSettingsStore((s) => s.showAll);
  const hideAll = useIndicatorSettingsStore((s) => s.hideAll);
  const shownCount = INDICATOR_DEFS.filter((d) => visible[d.key]).length;

  return (
    <div className="flex flex-col gap-2 text-[11px]">
      <div className="flex items-center justify-between shrink-0">
        <span className="text-[10px] text-[#444]">{shownCount}/{INDICATOR_DEFS.length} shown on the Dashboard</span>
        <div className="flex items-center gap-2">
          <button onClick={showAll} className="text-[10px] px-2 py-1 rounded border border-[#2a2a2a] text-[#666] hover:text-[#00ff7f] hover:border-[#00ff7f44] transition-colors">
            Show All
          </button>
          <button onClick={hideAll} className="text-[10px] px-2 py-1 rounded border border-[#2a2a2a] text-[#666] hover:text-[#ff3737] hover:border-[#ff373744] transition-colors">
            Hide All
          </button>
        </div>
      </div>
      <div className="flex flex-col">
        {INDICATOR_DEFS.map((d) => {
          const on = visible[d.key] ?? true;
          return (
            <label key={d.key} className="flex items-center justify-between gap-2 py-1.5 border-b border-[#161616] last:border-0 cursor-pointer">
              <div className="min-w-0">
                <div className="text-[#999] truncate">{d.label}</div>
                <div className="text-[10px] text-[#3a3a3a] truncate">{d.description}</div>
              </div>
              <button
                onClick={() => setVisible(d.key, !on)}
                className="relative w-9 h-5 rounded-full transition-colors shrink-0"
                style={{ background: on ? "#00ff7f33" : "#1a1a1a" }}
              >
                <span
                  className="absolute top-0.5 w-4 h-4 rounded-full transition-all"
                  style={{ left: on ? 18 : 2, background: on ? "#00ff7f" : "#555" }}
                />
              </button>
            </label>
          );
        })}
      </div>
      <div className="text-[10px] text-[#3a3a3a] leading-relaxed border-t border-[#161616] pt-2">
        Controls the Dashboard&apos;s &ldquo;Indicator Values&rdquo; panel. Saved to this browser
        (localStorage) — not synced across devices.
      </div>
    </div>
  );
}

// ── Section 3d — Account ────────────────────────────────────────────────────────

function AccountSection() {
  const { user, loading } = useCurrentUser();

  if (loading) return <div className="p-1"><SkeletonRows rows={2} /></div>;
  if (!user) return null;

  return (
    <div className="flex flex-col gap-3 text-[11px]">
      <ConfigRow label="Email" value={user.email} />
      <ConfigRow label="Display Name" value={user.display_name || "—"} />
      <Link
        href="/profile"
        className="flex items-center justify-center gap-1.5 py-1.5 rounded text-[10px] font-semibold uppercase tracking-wider bg-[#00ff7f]/10 text-[#00ff7f] border border-[#00ff7f]/30 hover:bg-[#00ff7f]/20 transition-colors"
      >
        <User size={11} /> Profile Settings
      </Link>
    </div>
  );
}

// ── Section 4 — System Status ──────────────────────────────────────────────────

function latencyColor(ms: number | null): string {
  if (ms == null) return "#555";
  if (ms < 200) return "#00ff7f";
  if (ms < 800) return "#ffd700";
  return "#ff3737";
}

function SystemStatusSection() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);

  useEffect(() => {
    const refresh = () => {
      api.getSystemStatus().then(setStatus).catch(() => setStatus(null));
      const t0 = performance.now();
      api.ping().then(() => setLatencyMs(Math.round(performance.now() - t0))).catch(() => setLatencyMs(null));
    };
    refresh();
    const id = setInterval(refresh, 15_000);
    return () => clearInterval(id);
  }, []);

  if (!status) return <div className="p-1"><SkeletonRows rows={4} /></div>;

  return (
    <div className="flex flex-col gap-3 text-[11px]">
      <div className="flex items-center justify-between">
        <span className="text-[#555]">GPU</span>
        {status.gpu.available ? (
          <span className="text-[#999] font-mono">
            {status.gpu.name} — {status.gpu.memory_used_mb}/{status.gpu.memory_total_mb} MB ({status.gpu.utilization_pct}%)
          </span>
        ) : (
          <Badge variant="neutral">{status.gpu.reason ?? "unavailable"}</Badge>
        )}
      </div>
      <div className="flex items-center justify-between gap-2">
        <span className="text-[#555]">Ollama</span>
        <Badge variant={status.ollama.connected ? "long" : "short"}>
          {status.ollama.connected ? `Connected · ${(status.ollama.models ?? []).join(", ") || "no models"}` : "Unreachable"}
        </Badge>
      </div>
      <div>
        <div className="flex items-center justify-between">
          <span className="text-[#555]">Redis</span>
          <Badge variant="neutral">
            {status.redis.connected ? "Connected (unused)" : status.redis.configured ? "Unreachable" : "Not configured"}
          </Badge>
        </div>
        <div className="text-[10px] text-[#3a3a3a] mt-0.5">
          Not load-bearing — scheduling runs on APScheduler, not Celery/Redis. This row is an
          optional health-check probe, never on the critical path.
        </div>
      </div>
      <div className="flex items-center justify-between">
        <span className="text-[#555]">Backend API Latency</span>
        <span className="font-mono" style={{ color: latencyColor(latencyMs) }}>{latencyMs != null ? `${latencyMs}ms` : "—"}</span>
      </div>
    </div>
  );
}

// ── Page ────────────────────────────────────────────────────────────────────

export default function SettingsPage() {
  const selectedAsset = useMARTAStore((s) => s.selectedAsset);
  const [freshness, setFreshness] = useState<Freshness | null>(null);

  useEffect(() => {
    fetchApi<Freshness>("/api/data/freshness").then(setFreshness).catch(() => {});
  }, []);

  return (
    <div className="flex flex-col gap-2 pb-4">
      <PageHeader title="Settings" description="Data sources, model configuration, agent behavior, and system status" />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        <Panel title="Account"><AccountSection /></Panel>
        <Panel title="Data Sources"><DataSourcesSection /></Panel>
        <Panel title="Model Configuration"><ModelConfigSection /></Panel>
        <Panel title="Agent Configuration" actions={<Badge variant="warning">Not wired</Badge>}><AgentConfigSection /></Panel>
        <Panel title="Execution Controls"><ExecutionControlsSection /></Panel>
        <Panel title="RL Execution"><RlExecutionSection /></Panel>
        <Panel title="Weekly PPO"><WeeklyPpoSection /></Panel>
        <Panel title="Indicator Visibility"><IndicatorVisibilitySection /></Panel>
        <Panel title="System Status"><SystemStatusSection /></Panel>
      </div>

      <div className="grid grid-cols-2 gap-2 shrink-0">
        <Panel title="Connection">
          <ConfigRow label="API Base URL" value={API_URL} />
          <ConfigRow label="WebSocket URL" value={WS_URL} />
          <ConfigRow label="Selected Asset" value={selectedAsset} />
        </Panel>
        <Panel title="Architecture">
          <div className="flex flex-col gap-2 text-[11px]">
            <div className="flex items-center justify-between">
              <span className="text-[#555]">Pipeline</span>
              <span className="text-[#999] font-mono text-[10px]">TFT + InfoNCE → HMM → 5-agent debate → PPO</span>
            </div>
          </div>
        </Panel>
      </div>

      <Panel title="Data Freshness">
        {!freshness ? (
          <Spinner label="Loading freshness report…" />
        ) : (
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-3 text-[11px] text-[#555] shrink-0">
              <span>Checked {new Date(freshness.checked_at).toLocaleString()}</span>
              {freshness.cron_last_ran && <span>· Cron last ran {new Date(freshness.cron_last_ran).toLocaleString()}</span>}
              {freshness.any_stale ? <Badge variant="warning">Stale data detected</Badge> : <Badge variant="long">All fresh</Badge>}
            </div>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Asset</th>
                  <th>Type</th>
                  <th>Last Timestamp</th>
                  <th>Age (h)</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {freshness.assets.map((a) => (
                  <tr key={a.id}>
                    <td>{a.label}</td>
                    <td className="text-[#555] uppercase">{a.type}</td>
                    <td className="font-mono text-[#777]">{a.last_timestamp ? new Date(a.last_timestamp).toLocaleString() : "—"}</td>
                    <td className="font-mono">{a.age_hours}</td>
                    <td>{a.stale ? <Badge variant="warning">Stale</Badge> : <Badge variant="long">Fresh</Badge>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <LearningPanel>
        The encoder uses InfoNCE contrastive loss — it learns to make similar market windows cluster
        together in embedding space, and dissimilar windows push apart, without any labels. The regime
        HMM and everything downstream (agents, PPO) only ever sees that learned 128-dim fingerprint,
        never the raw price/indicator windows directly.
      </LearningPanel>
    </div>
  );
}
