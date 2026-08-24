"""
Month 2 Week 2 demo — run 4 debates across different regime states and log to MLflow.

Crypto-only panel (2026-08-23 pivot) — 5 assets (BTC/ETH/BNB/LTC/XRP_USDT), regime
semantics per data/validated/regime_semantics_crypto5_multihorizon.json (n_states=6,
every asset has rows in every regime — unlike the old 4-asset panel, no per-asset
regime-availability gaps to worry about):
    - BTC Regime 4 (high-vol bull, global Sharpe 1.39, ann. ret +141%)
    - BTC Regime 0 (bear/crash, global Sharpe -1.04, ann. ret -35%)
    - XRP Regime 5 (high-vol bear/stress — highest ann. vol of any regime at 190%,
      global Sharpe -0.44 — replaces the old panel's GOLD stress/crisis scenario)
    - BTC Regime 4 with high conviction (regime_prob > 0.80) — designed to trigger Contrarian

Usage:
    python run_debate.py                         # 4 demo debates
    python run_debate.py --val-window            # sample across val window (24h stride)
    python run_debate.py --val-window --stride 48  # every 48h instead
    python run_debate.py --val-window --append   # append to existing debate_results.json
"""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import mlflow
import pandas as pd
from rich.console import Console
from rich.table import Table

from agents.contrarian_agent import ContrarianAgent
from agents.fund_manager import FundManagerAgent
from agents.graph import DebateGraph, run_debate as _run_debate_raw
from agents.macro_agent import MacroAgent
from agents.risk_agent import RiskAgent
from agents.schemas import DebateResult
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from utils.freshness import CRYPTO5_DEBATE_ASSETS, CRYPTO5_PROCESSED, CRYPTO5_SEMANTICS_PATH, load_debate_data
from utils.logger import logger

_DATA_DIR = CRYPTO5_PROCESSED
_DEBATE_ASSETS = CRYPTO5_DEBATE_ASSETS
_SEMANTICS_PATH = CRYPTO5_SEMANTICS_PATH
_OUTPUT_DIR = Path("data/debates")
_MLFLOW_EXPERIMENT = "marta_agent_deliberation"

# Every run_debate(...) call in this file must pass the same semantics_path used to
# build FundManagerAgent below, or the displayed regime description and the
# conviction/override logic will disagree about which regime stream's semantics apply
# (agents/graph.py's run_debate docstring). Wrapped once here so every call site below
# gets it automatically instead of repeating the kwarg at each of the 3 call sites.
run_debate = partial(_run_debate_raw, semantics_path=_SEMANTICS_PATH)


def _load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Regime stream + per-asset aligned parquets, via the shared loader (P0.4),
    explicitly requesting the crypto-only 5-asset panel (data/processed/
    crypto5_multihorizon) — load_debate_data()'s own default is still the old 4-asset
    production panel, since dashboard/backend/debate_stream.py and the other three
    debate-generation scripts rely on that default being unchanged. Was previously a
    bespoke, unchecked implementation duplicated near-verbatim across all four
    debate-generation scripts."""
    return load_debate_data(processed_dir=_DATA_DIR, assets=_DEBATE_ASSETS)


def _select_demo_timestamps(
    regime_stream: pd.DataFrame,
) -> list[tuple[pd.Timestamp, str, int, str]]:
    """
    Returns (timestamp, asset, regime_label, description) for 4 debates.
    Timestamps enforced to be at least 30 days apart.
    Debate 4 selects BTC regime_1 with highest regime_prob to guarantee Contrarian activation.
    """
    stream = regime_stream.copy()
    stream["timestamp"] = pd.to_datetime(stream["timestamp"])

    def _pick(
        asset: str,
        label: int,
        after: str | None = None,
        before: str | None = None,
    ) -> pd.Timestamp | None:
        rows = stream[
            (stream["asset"] == asset) & (stream["regime_label"] == label)
        ].reset_index(drop=True)
        valid = rows[rows.index > 200]
        if valid.empty:
            valid = rows[rows.index > 60]
        if valid.empty:
            return None
        if after:
            narrow = valid[valid["timestamp"] >= pd.Timestamp(after, tz="UTC")]
            if not narrow.empty:
                valid = narrow
        if before:
            narrow = valid[valid["timestamp"] < pd.Timestamp(before, tz="UTC")]
            if not narrow.empty:
                valid = narrow
        return valid.iloc[min(500, len(valid) - 1)]["timestamp"]

    def _pick_high_prob(
        asset: str, label: int, min_prob: float = 0.80
    ) -> pd.Timestamp | None:
        """Pick timestamp with highest regime probability for the given label — maximises Contrarian activation."""
        rows = stream[
            (stream["asset"] == asset) & (stream["regime_label"] == label)
        ].copy().reset_index(drop=True)
        prob_col = f"regime_prob_{label}"
        if prob_col in rows.columns:
            rows = rows[rows[prob_col] >= min_prob]
        if rows.empty:
            return _pick(asset, label)
        # Sort by probability descending, skip early rows for warmup
        if prob_col in rows.columns:
            rows = rows.sort_values(prob_col, ascending=False)
        valid = rows[rows.index > 50] if len(rows) > 100 else rows
        return valid.iloc[0]["timestamp"] if not valid.empty else rows.iloc[0]["timestamp"]

    scenarios = [
        ("BTC_USDT", 4, "BTC high-vol bull", None, None, False),
        ("BTC_USDT", 0, "BTC bear/crash", None, None, False),
        ("XRP_USDT", 5, "XRP high-vol bear/stress", None, None, False),
        ("BTC_USDT", 4, "BTC high-conviction bull [Contrarian trigger]", None, None, True),
    ]

    picks: list[tuple[pd.Timestamp, str, int, str]] = []

    for asset, label, desc, after, before, high_prob in scenarios:
        ts = _pick_high_prob(asset, label) if high_prob else _pick(asset, label, after, before)
        if ts is None:
            logger.warning(f"No valid rows for {asset} regime {label} — skipping")
            continue

        # Enforce 30-day separation from all previously selected timestamps
        too_close = [p for p, _, _, _ in picks if abs((ts - p).days) < 30]
        if too_close:
            rows = stream[
                (stream["asset"] == asset) & (stream["regime_label"] == label)
            ].reset_index(drop=True)
            valid = rows[rows.index > 200] if not high_prob else rows
            if valid.empty:
                valid = rows[rows.index > 60]
            found = None
            for candidate in valid["timestamp"]:
                if all(abs((candidate - p).days) >= 30 for p, _, _, _ in picks):
                    found = candidate
                    break
            if found is not None:
                ts = found
            else:
                logger.warning(f"{desc} @ {ts} is <30 days from another pick — no alternative found, using closest")

        picks.append((ts, asset, label, desc))
        logger.info(f"Selected {desc}: {ts}")

    return picks


def _resolve_date_timestamp(
    regime_stream: pd.DataFrame, asset: str, date_str: str
) -> pd.Timestamp:
    """Nearest regime_stream row for `asset` to the requested date/timestamp.

    regime_stream.parquet is a static snapshot that stops growing the moment
    run_regime.py last ran — a requested date past its coverage silently snaps to
    the nearest edge row rather than erroring, so this always warns loudly when
    the resolved timestamp is more than a few days from what was asked for.
    """
    stream = regime_stream[regime_stream["asset"] == asset].copy()
    if stream.empty:
        raise ValueError(f"No regime_stream rows for asset '{asset}'")
    stream["timestamp"] = pd.to_datetime(stream["timestamp"], utc=True)
    target = pd.Timestamp(date_str, tz="UTC")

    idx = (stream["timestamp"] - target).abs().idxmin()
    resolved = pd.Timestamp(stream.loc[idx, "timestamp"])
    delta_days = abs((resolved - target).days)
    if delta_days > 3:
        lo, hi = stream["timestamp"].min(), stream["timestamp"].max()
        logger.warning(
            f"Requested date {date_str} is {delta_days} days from the nearest available "
            f"{asset} row — regime_stream.parquet only covers {lo.date()} .. {hi.date()}. "
            f"Falling back to nearest available timestamp: {resolved}"
        )
    return resolved


def _print_debate_transcript(desc: str, result: DebateResult, console: Console) -> None:
    console.rule(f"[bold cyan]{desc}[/bold cyan]")
    console.print(
        f"  Asset: [bold]{result.asset}[/bold] | Regime: R{result.regime_label} | "
        f"Timestamp: {result.timestamp[:19]}"
    )

    console.print("\n  [bold yellow]Round 1 Theses:[/bold yellow]")
    for t in result.round1_theses:
        if t.agent_name == "ContrarianAgent":
            continue  # print separately below
        bias_color = {"bullish": "green", "bearish": "red", "neutral": "yellow"}.get(
            t.directional_bias, "white"
        )
        console.print(
            f"    [{t.agent_name}] [{bias_color}]{t.directional_bias}[/{bias_color}] "
            f"conf={t.confidence:.2f} align={t.regime_alignment}"
        )
        console.print(f"      {t.reasoning[:200]}")

    # Print Contrarian thesis with distinctive label
    contrarian_theses = [t for t in result.round1_theses if t.agent_name == "ContrarianAgent"]
    if contrarian_theses:
        ct = contrarian_theses[0]
        if ct.directional_bias != "neutral":
            bias_color = {"bullish": "green", "bearish": "red", "neutral": "yellow"}.get(
                ct.directional_bias, "white"
            )
            console.print(
                f"\n  [bold red][CONTRARIAN — Devil's Advocate][/bold red]"
                f"    [{bias_color}]{ct.directional_bias}[/{bias_color}] "
                f"conf={ct.confidence:.2f} align={ct.regime_alignment}"
            )
            console.print(f"      {ct.reasoning[:200]}")
        else:
            console.print(
                f"\n  [dim][CONTRARIAN — inactive (consensus confidence too low)][/dim]"
            )

    console.print("\n  [bold yellow]Round 2 Rebuttals:[/bold yellow]")
    if result.round2_rebuttals:
        for r in result.round2_rebuttals:
            name_color = "red" if r.agent_name == "ContrarianAgent" else "cyan"
            console.print(
                f"    [[{name_color}]{r.agent_name}[/{name_color}]] → [{r.target_agent}] | "
                f"maintains={r.maintains_bias} conf={r.updated_confidence:.2f}"
            )
            console.print(f"      {r.rebuttal_point[:200]}")
    else:
        console.print("    No rebuttals posted.")

    fm = result.fund_manager_decision
    action_color = {"long": "green", "short": "red", "hold": "yellow"}.get(fm.final_action, "white")
    console.print(
        f"\n  [bold]Risk:[/bold] {result.risk_limits.risk_regime} | "
        f"vol_forecast={result.risk_limits.volatility_forecast:.4f} | "
        f"max_pos={result.risk_limits.max_position_size:.3f}"
    )
    dissenting_str = f" | dissenting={fm.dissenting_agents}" if fm.dissenting_agents else ""
    penalty_str = f" | penalty=[red]{fm.regime_penalty_applied:.2f}x[/red]" if fm.regime_penalty_applied < 1.0 else f" | penalty={fm.regime_penalty_applied:.2f}x"
    contradiction_str = " | [red]REGIME CONTRADICTION[/red]" if fm.regime_contradiction else ""
    console.print(
        f"\n  [bold]Final Decision:[/bold] [{action_color}]{fm.final_action.upper()}[/{action_color}] "
        f"size={fm.position_size:.3f} conviction={fm.conviction:.4f} "
        f"override={'[red]YES[/red]' if fm.regime_override else 'no'}"
        f"{penalty_str}{contradiction_str}{dissenting_str}"
    )
    console.print(f"  [italic]{fm.rationale[:300]}[/italic]")
    console.print(f"  [dim]Tokens: {result.total_tokens_used} | Duration: {result.debate_duration_seconds}s[/dim]\n")


def _print_summary(results: list[tuple[str, DebateResult]], console: Console) -> None:
    table = Table(title="MARTA Debate Summary — Month 2 Week 3 (Regime-Conditioned Conviction)", show_lines=True)
    table.add_column("Scenario", style="cyan", min_width=28)
    table.add_column("Timestamp", style="dim")
    table.add_column("Regime", style="magenta")
    table.add_column("Consensus", style="yellow")
    table.add_column("Action", style="bold")
    table.add_column("Conv.", justify="right")
    table.add_column("Penalty", justify="right")
    table.add_column("Contradict", justify="center")
    table.add_column("Contrarian", justify="center")
    table.add_column("Rebuttals", justify="center")
    table.add_column("Override", justify="center")
    table.add_column("Dur(s)", justify="right")

    for desc, r in results:
        fm = r.fund_manager_decision
        action_color = {"long": "green", "short": "red", "hold": "yellow"}.get(fm.final_action, "white")
        contrarian_theses = [t for t in r.round1_theses if t.agent_name == "ContrarianAgent"]
        contrarian_active = bool(contrarian_theses and contrarian_theses[0].directional_bias != "neutral")
        penalty_color = "red" if fm.regime_penalty_applied < 0.80 else ("yellow" if fm.regime_penalty_applied < 1.0 else "green")
        table.add_row(
            desc[:28],
            r.timestamp[:16],
            f"R{r.regime_label}",
            fm.consensus_direction,
            f"[{action_color}]{fm.final_action}[/{action_color}]",
            f"{fm.conviction:.3f}",
            f"[{penalty_color}]{fm.regime_penalty_applied:.2f}x[/{penalty_color}]",
            "[red]YES[/red]" if fm.regime_contradiction else "[dim]—[/dim]",
            "[red]ACTIVE[/red]" if contrarian_active else "[dim]—[/dim]",
            str(len(r.round2_rebuttals)),
            "[red]YES[/red]" if fm.regime_override else "no",
            str(r.debate_duration_seconds),
        )

    console.print(table)


def _select_val_window_timestamps(
    regime_stream: pd.DataFrame,
    stride_hours: int = 24,
) -> list[tuple[pd.Timestamp, str, int, str]]:
    """Sample BTC timestamps at `stride_hours` intervals across the 2021 val window
    (2021-01-01 → 2021-10-12) so the consensus slot is non-zero for ablation purposes.

    Only timestamps with a valid regime row AND enough aligned history (>200 rows) are kept.
    """
    val_start = pd.Timestamp("2021-01-01", tz="UTC")
    val_end   = pd.Timestamp("2021-10-12", tz="UTC")

    stream = regime_stream.copy()
    stream["timestamp"] = pd.to_datetime(stream["timestamp"], utc=True)

    btc_rows = stream[
        (stream["asset"] == "BTC_USDT")
        & (stream["timestamp"] >= val_start)
        & (stream["timestamp"] <= val_end)
    ].sort_values("timestamp").reset_index(drop=True)

    if btc_rows.empty:
        logger.warning("No BTC_USDT rows in val window — cannot sample val-window debates")
        return []

    # Pick one row every `stride_hours` hours
    picks: list[tuple[pd.Timestamp, str, int, str]] = []
    last_ts: pd.Timestamp | None = None
    for _, row in btc_rows.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        if last_ts is None or (ts - last_ts).total_seconds() >= stride_hours * 3600:
            label = int(row["regime_label"])
            desc = f"BTC val-window R{label} @ {str(ts)[:10]}"
            picks.append((ts, "BTC_USDT", label, desc))
            last_ts = ts

    logger.info(f"Val-window sampling: {len(picks)} timestamps (stride={stride_hours}h)")
    return picks


def _run_val_window(
    regime_stream: pd.DataFrame,
    aligned_dfs: dict[str, pd.DataFrame],
    debate_graph: "DebateGraph",
    stride_hours: int,
    append: bool,
    console: Console,
) -> None:
    """Run debates for the full 2021 val window and save to debate_results.json."""
    timestamps = _select_val_window_timestamps(regime_stream, stride_hours)
    if not timestamps:
        console.print("[yellow]No valid timestamps found in val window — nothing to run.[/yellow]")
        return

    output_path = _OUTPUT_DIR / "debate_results.json"
    existing: list[dict] = []
    if append and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        console.print(f"[dim]Appending to {len(existing)} existing debates in {output_path}[/dim]")

    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    serialized = list(existing)

    with mlflow.start_run(run_name=f"val_window_debates_s{stride_hours}h") as run:
        mlflow.log_param("n_timestamps", len(timestamps))
        mlflow.log_param("stride_hours", stride_hours)
        mlflow.log_param("val_start", "2021-01-01")
        mlflow.log_param("val_end", "2021-10-12")

        for i, (ts, asset, label, desc) in enumerate(timestamps):
            console.print(
                f"[bold]Val-window debate {i+1}/{len(timestamps)}:[/bold] {desc}"
            )
            try:
                result = run_debate(ts, asset, regime_stream, aligned_dfs, debate_graph)
                d = result.model_dump()
                d["scenario_description"] = desc
                serialized.append(d)
                fm = result.fund_manager_decision
                mlflow.log_metrics({
                    f"debate_{i+1}_conviction": fm.conviction,
                    f"debate_{i+1}_regime": label,
                }, step=i)
            except Exception as exc:
                logger.warning(f"Debate {i+1} failed ({exc}) — skipping")
                continue

        with open(output_path, "w") as f:
            json.dump(serialized, f, indent=2, default=str)
        mlflow.log_artifact(str(output_path))

    total_new = len(serialized) - len(existing)
    console.print(
        f"[bold green]Val-window complete:[/bold green] "
        f"{total_new} new debates saved → {output_path} "
        f"({len(serialized)} total)"
    )
    console.print(f"[bold]MLflow run:[/bold] http://127.0.0.1:5000")


def _run_single_date(
    date_str: str,
    asset: str,
    regime_stream: pd.DataFrame,
    aligned_dfs: dict[str, pd.DataFrame],
    debate_graph: "DebateGraph",
    append: bool,
    console: Console,
) -> None:
    """Run one debate for the timestamp nearest `date_str`, print the transcript,
    and save/append it to debate_results.json (same file the other modes write)."""
    ts = _resolve_date_timestamp(regime_stream, asset, date_str)
    desc = f"{asset} @ {date_str} (resolved: {str(ts)[:16]})"
    console.print(f"[bold]Running single debate:[/bold] {desc}")
    logger.info(f"Single-date debate: {desc}")

    output_path = _OUTPUT_DIR / "debate_results.json"
    existing: list[dict] = []
    if append and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        console.print(f"[dim]Appending to {len(existing)} existing debates in {output_path}[/dim]")

    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"single_date_{asset}_{date_str}") as run:
        mlflow.log_param("asset", asset)
        mlflow.log_param("requested_date", date_str)
        mlflow.log_param("resolved_timestamp", str(ts))

        result = run_debate(ts, asset, regime_stream, aligned_dfs, debate_graph)
        _print_debate_transcript(desc, result, console)

        fm = result.fund_manager_decision
        mlflow.log_metrics({
            "regime_label": result.regime_label,
            "conviction": fm.conviction,
            "position_size": fm.position_size,
            "regime_override": int(fm.regime_override),
            "tokens": result.total_tokens_used,
            "duration_s": result.debate_duration_seconds,
        })
        mlflow.log_param("action", fm.final_action)
        mlflow.log_param("consensus", fm.consensus_direction)

        d = result.model_dump()
        d["scenario_description"] = desc
        serialized = existing + [d]

        with open(output_path, "w") as f:
            json.dump(serialized, f, indent=2, default=str)
        mlflow.log_artifact(str(output_path))

        mlflow_url = f"http://127.0.0.1:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
        console.print(f"\n[bold green]Saved:[/bold green] {output_path} ({len(serialized)} debates total)")
        console.print(f"[bold green]MLflow run:[/bold green] {mlflow_url}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="MARTA debate runner")
    parser.add_argument(
        "--val-window",
        action="store_true",
        help="sample debates across the full 2021 validation window (needed for ablation)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=24,
        metavar="HOURS",
        help="hours between sampled timestamps in val-window mode (default: 24)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="append to existing debate_results.json instead of overwriting",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="run a single debate for the date/timestamp nearest this value "
             "(regime_stream.parquet coverage only — see CLAUDE.md for current bounds)",
    )
    parser.add_argument(
        "--asset",
        type=str,
        default="BTC_USDT",
        choices=_DEBATE_ASSETS,
        help="asset for --date mode (default: BTC_USDT)",
    )
    args = parser.parse_args()

    console = Console()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    console.rule("[bold magenta]MARTA Month 2 Week 2 — 5-Agent Debate Demo[/bold magenta]")

    logger.info("Loading data...")
    regime_stream, aligned_dfs = _load_data()
    logger.info(f"Loaded regime stream: {len(regime_stream)} rows | {len(aligned_dfs)} aligned assets")

    logger.info("Initializing agents (crypto-only panel)...")
    macro_agent = MacroAgent(assets=_DEBATE_ASSETS, data_dir=_DATA_DIR)
    tech_agent = TechnicalAgent()
    risk_agent = RiskAgent(data_dir=_DATA_DIR)
    fm_agent = FundManagerAgent(semantics_path=_SEMANTICS_PATH)
    sentiment_agent = SentimentAgent(
        macro_assets={"macro"}, fear_greed_assets=set(_DEBATE_ASSETS)
    )
    contrarian_agent = ContrarianAgent(regime_stream_df=regime_stream)
    debate_graph = DebateGraph(
        macro_agent, tech_agent, risk_agent, fm_agent, sentiment_agent, contrarian_agent
    )
    logger.info("All 5 agents + RiskAgent initialized")

    # Val-window mode: populate the debate cache for the full 2021 validation period.
    if args.val_window:
        _run_val_window(regime_stream, aligned_dfs, debate_graph, args.stride, args.append, console)
        return

    # Single-date mode: one debate for the timestamp nearest the requested date.
    if args.date:
        _run_single_date(
            args.date, args.asset, regime_stream, aligned_dfs, debate_graph, args.append, console
        )
        return

    demo_timestamps = _select_demo_timestamps(regime_stream)
    logger.info(f"Running {len(demo_timestamps)} debates...")

    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    all_results: list[tuple[str, DebateResult]] = []

    with mlflow.start_run(run_name="week2_demo_4debates") as run:
        mlflow.log_param("n_debates", len(demo_timestamps))
        mlflow.log_param("model", "mistral:7b-instruct")
        mlflow.log_param("agents", "MacroAgent,TechnicalAgent,SentimentAgent,ContrarianAgent,RiskAgent,FundManager")
        mlflow.log_param("week", "2")

        for i, (ts, asset, label, desc) in enumerate(demo_timestamps):
            console.print(f"\n[bold]Running debate {i+1}/{len(demo_timestamps)}: {desc}[/bold]")
            logger.info(f"Debate {i+1}: {desc} @ {ts}")

            result = run_debate(ts, asset, regime_stream, aligned_dfs, debate_graph)
            all_results.append((desc, result))

            _print_debate_transcript(desc, result, console)

            prefix = f"debate_{i+1}"
            fm = result.fund_manager_decision
            contrarian_theses = [t for t in result.round1_theses if t.agent_name == "ContrarianAgent"]
            contrarian_active = bool(contrarian_theses and contrarian_theses[0].directional_bias != "neutral")

            mlflow.log_metrics({
                f"{prefix}_regime_label": result.regime_label,
                f"{prefix}_conviction": fm.conviction,
                f"{prefix}_position_size": fm.position_size,
                f"{prefix}_regime_override": int(fm.regime_override),
                f"{prefix}_tokens": result.total_tokens_used,
                f"{prefix}_duration_s": result.debate_duration_seconds,
                f"{prefix}_vol_forecast": result.risk_limits.volatility_forecast,
                f"{prefix}_n_rebuttals": len(result.round2_rebuttals),
                f"{prefix}_contrarian_active": int(contrarian_active),
            })
            mlflow.log_param(f"{prefix}_asset", result.asset)
            mlflow.log_param(f"{prefix}_action", fm.final_action)
            mlflow.log_param(f"{prefix}_consensus", fm.consensus_direction)

        total_tokens = sum(r.total_tokens_used for _, r in all_results)
        total_duration = sum(r.debate_duration_seconds for _, r in all_results)
        total_rebuttals = sum(len(r.round2_rebuttals) for _, r in all_results)
        contrarian_activations = sum(
            1 for _, r in all_results
            if any(t.agent_name == "ContrarianAgent" and t.directional_bias != "neutral"
                   for t in r.round1_theses)
        )
        mlflow.log_metrics({
            "total_tokens": total_tokens,
            "total_duration_s": total_duration,
            "avg_conviction": sum(r.fund_manager_decision.conviction for _, r in all_results) / len(all_results),
            "total_rebuttals_fired": total_rebuttals,
            "contrarian_activations": contrarian_activations,
        })

        output_path = _OUTPUT_DIR / "debate_results.json"
        serialized = []
        for desc, r in all_results:
            d = r.model_dump()
            d["scenario_description"] = desc
            serialized.append(d)

        with open(output_path, "w") as f:
            json.dump(serialized, f, indent=2, default=str)

        mlflow.log_artifact(str(output_path))
        logger.success(f"Debate results saved to {output_path}")

        console.print()
        _print_summary(all_results, console)

        mlflow_url = f"http://127.0.0.1:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
        console.print(f"\n[bold green]MLflow run:[/bold green] {mlflow_url}")
        console.print(f"[bold green]Results saved:[/bold green] {output_path}")

        actions = [r.fund_manager_decision.final_action for _, r in all_results]
        convictions = [r.fund_manager_decision.conviction for _, r in all_results]
        console.print(f"\n[bold]Actions across regimes:[/bold] {actions}")
        console.print(f"[bold]Convictions:[/bold] {[f'{c:.2f}' for c in convictions]}")
        console.print(f"[bold]Contrarian activations:[/bold] {contrarian_activations}/{len(all_results)}")
        console.print(f"[bold]Total Round 2 rebuttals:[/bold] {total_rebuttals}")

        conviction_spread = max(convictions) - min(convictions)
        varied = len(set(actions)) > 1 or conviction_spread > 0.05
        if varied:
            console.print(
                f"[bold green]✓ Regime conditioning produces different decisions "
                f"(conviction spread: {conviction_spread:.3f})[/bold green]"
            )
        else:
            console.print("[yellow]⚠ All debates produced identical actions — check LLM responses[/yellow]")

        if contrarian_activations >= 1:
            console.print("[bold green]✓ Contrarian activated in at least 1 debate[/bold green]")
        else:
            console.print("[yellow]⚠ Contrarian did not activate — check consensus confidence levels[/yellow]")

        if total_rebuttals > 0:
            console.print("[bold green]✓ Round 2 rebuttals fired in at least 1 debate[/bold green]")
        else:
            console.print("[yellow]⚠ No Round 2 rebuttals fired[/yellow]")


if __name__ == "__main__":
    main()
