import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

logger = logging.getLogger(__name__)

MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).parent.parent.parent)))

# LangGraph's .invoke() (called inside DebateGraph.run) is synchronous and calls Ollama
# over blocking HTTP — running it directly on the asyncio event loop freezes every other
# request (REST + other websockets) for the full ~30-90s debate. max_workers=1 because
# the underlying Ollama GPU inference is effectively serialized anyway; this just keeps
# the event loop free while it runs rather than promising concurrency the GPU can't give.
_debate_executor = ThreadPoolExecutor(max_workers=1)

_debates: list[dict] | None = None


_DEBATES_PATH = MARTA_ROOT / "data" / "debates" / "debate_results.json"


def _load_debates() -> list[dict]:
    global _debates
    if _debates is None:
        if _DEBATES_PATH.exists():
            with open(_DEBATES_PATH) as f:
                _debates = json.load(f)
        else:
            _debates = []
    return _debates


def _save_debate_result(result: dict) -> None:
    """Append a completed live debate to debate_results.json and keep the in-memory
    cache in sync, so /api/debates reflects it immediately without a server restart.

    Builds the new list from a *copy* of the cached one and only swaps the module-level
    `_debates` reference in after a successful write — `_load_debates()` returns the
    live cached list by reference, so appending to it directly (then having the write
    fail, e.g. a read-only mount) would leave the in-memory list holding a debate that
    was never actually persisted to disk.
    """
    try:
        existing = list(_load_debates())
        existing.append(result)
        _DEBATES_PATH.write_text(json.dumps(existing, indent=2, default=str))
    except Exception:
        logger.warning("Could not save live debate result", exc_info=True)
        return
    global _debates
    _debates = existing


def _thesis_to_msg(thesis: dict) -> dict:
    return {
        "type": "agent_thesis",
        "agent": thesis.get("agent_name", ""),
        "bias": thesis.get("directional_bias", "neutral"),
        "confidence": thesis.get("confidence", 0.0),
        "reasoning": thesis.get("reasoning", ""),
        "key_factors": thesis.get("key_factors", []),
        "regime_alignment": thesis.get("regime_alignment", "neutral"),
    }


def _rebuttal_to_msg(r: dict) -> dict:
    return {
        "type": "agent_rebuttal",
        "agent": r.get("agent_name", ""),
        "target": r.get("target_agent", ""),
        "rebuttal_point": r.get("rebuttal_point", ""),
        "maintains_bias": r.get("maintains_bias", True),
        "updated_confidence": r.get("updated_confidence", 0.5),
    }


async def stream_debate_replay(websocket: WebSocket, debate_index: int) -> None:
    try:
        await _stream_debate_replay(websocket, debate_index)
    except WebSocketDisconnect:
        pass


async def _stream_debate_replay(websocket: WebSocket, debate_index: int) -> None:
    debates = _load_debates()
    if not debates or debate_index >= len(debates):
        await websocket.send_json({"type": "error", "message": "Debate not found"})
        return

    debate = debates[debate_index]
    theses: list[dict] = debate.get("round1_theses", [])
    rebuttals: list[dict] = debate.get("round2_rebuttals", [])
    risk: dict = debate.get("risk_limits", {})
    fm: dict = debate.get("fund_manager_decision", {})

    await websocket.send_json({
        "type": "debate_start",
        "asset": debate.get("asset"),
        "timestamp": debate.get("timestamp"),
        "regime_label": debate.get("regime_label"),
        "scenario": debate.get("scenario_description", ""),
    })
    await asyncio.sleep(0.5)

    await websocket.send_json({"type": "round1_start"})
    await asyncio.sleep(0.3)

    # Separate contrarian from R1 agents
    r1_regular = [t for t in theses if t.get("agent_name") != "ContrarianAgent"]
    r1_contrarian = [t for t in theses if t.get("agent_name") == "ContrarianAgent"]

    for thesis in r1_regular:
        await websocket.send_json({"type": "agent_thinking", "agent": thesis.get("agent_name", "")})
        await asyncio.sleep(0.6)
        await websocket.send_json(_thesis_to_msg(thesis))
        await asyncio.sleep(0.8)

    # Risk limits (from GARCH, no LLM)
    await websocket.send_json({
        "type": "risk_limits",
        "max_position_size": risk.get("max_position_size", 0),
        "volatility_forecast": risk.get("volatility_forecast", 0),
        "cvar_95": risk.get("cvar_95", 0),
        "risk_regime": risk.get("risk_regime", "medium"),
    })
    await asyncio.sleep(0.5)

    # Contrarian (if present)
    if r1_contrarian:
        contrarian = r1_contrarian[0]
        avg_conf = sum(t.get("confidence", 0) for t in r1_regular) / max(len(r1_regular), 1)
        await websocket.send_json({
            "type": "contrarian_activated",
            "avg_confidence": round(avg_conf, 3),
        })
        await asyncio.sleep(0.4)
        await websocket.send_json({"type": "agent_thinking", "agent": "ContrarianAgent"})
        await asyncio.sleep(0.6)
        await websocket.send_json(_thesis_to_msg(contrarian))
        await asyncio.sleep(0.8)

    # Round 2
    if rebuttals:
        await websocket.send_json({"type": "round2_start"})
        await asyncio.sleep(0.3)
        for r in rebuttals:
            await websocket.send_json(_rebuttal_to_msg(r))
            await asyncio.sleep(0.4)

    # Fund Manager
    await websocket.send_json({"type": "agent_thinking", "agent": "FundManagerAgent"})
    await asyncio.sleep(0.8)
    await websocket.send_json({
        "type": "fund_manager",
        "action": fm.get("final_action", "hold"),
        "position_size": fm.get("position_size", 0),
        "conviction": fm.get("conviction", 0),
        "rationale": fm.get("rationale", ""),
        "regime_override": fm.get("regime_override", False),
        "dissenting_agents": fm.get("dissenting_agents", []),
    })
    await asyncio.sleep(0.5)

    # PPO stub (use FM decision as proxy)
    await websocket.send_json({
        "type": "ppo_decision",
        "action": fm.get("final_action", "hold"),
        "position_size": fm.get("position_size", 0) * fm.get("regime_penalty_applied", 1.0),
        "regime_override": fm.get("regime_override", False),
    })
    await asyncio.sleep(0.3)

    await websocket.send_json({"type": "debate_complete"})


async def stream_debate_live(websocket: WebSocket, asset: str, timestamp: str) -> None:
    """Run a real debate via the MARTA agent system."""
    import sys
    sys.path.insert(0, str(MARTA_ROOT))
    try:
        from agents.graph import DebateGraph
        from agents.context import build_regime_context
        from agents.macro_agent import MacroAgent
        from agents.technical_agent import TechnicalAgent
        from agents.risk_agent import RiskAgent
        from agents.fund_manager import FundManagerAgent
        from agents.sentiment_agent import SentimentAgent
        from agents.contrarian_agent import ContrarianAgent
        from utils.assets import logical_asset_name
        import pandas as pd

        await websocket.send_json({"type": "debate_start", "asset": asset, "timestamp": timestamp, "mode": "live"})

        ts = pd.Timestamp(timestamp, tz="UTC")

        regime_stream_df = pd.read_parquet(
            MARTA_ROOT / "data/processed/regime_stream.parquet", engine="pyarrow"
        )
        aligned_full = pd.read_parquet(
            MARTA_ROOT / "data/processed/all_assets_aligned.parquet", engine="pyarrow"
        )
        # Build per-asset dict and single-asset df for context. all_assets_aligned.parquet's
        # MultiIndex top-level columns are keyed by file stem (GOLD -> "XAUUSD", its histdata
        # instrument name), but `asset` here and every downstream agent lookup
        # (agents/graph.py's aligned_dfs.get(ctx.asset)) use the logical name ("GOLD") — hence
        # the KeyError('GOLD') this crashed with. Translate the dict's keys back to logical
        # names once, then look the single-asset df up from that already-translated dict.
        aligned_dfs = {
            logical_asset_name(col): aligned_full[col]
            for col in aligned_full.columns.get_level_values(0).unique()
        }
        aligned_df = aligned_dfs.get(asset)

        context = build_regime_context(ts, asset, regime_stream_df, aligned_df)

        graph = DebateGraph(
            MacroAgent(), TechnicalAgent(), RiskAgent(),
            FundManagerAgent(), SentimentAgent(), ContrarianAgent(),
        )

        # Stream R1 thinking indicators
        for agent_name in ["MacroAgent", "TechnicalAgent", "SentimentAgent"]:
            await websocket.send_json({"type": "agent_thinking", "agent": agent_name})
            await asyncio.sleep(0.2)

        # graph.run() invokes LangGraph's synchronous .invoke(), which makes blocking
        # HTTP calls to Ollama — offload to a worker thread so this ~30-90s debate
        # doesn't freeze the event loop for every other request in the meantime.
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_debate_executor, graph.run, context, aligned_dfs)
        theses = result.round1_theses
        rebuttals = result.round2_rebuttals
        fm = result.fund_manager_decision
        risk = result.risk_limits

        # Persist the live debate the same way the offline generation scripts do, so it
        # shows up in the debate list/history immediately, not just for this one replay.
        result_dict = result.model_dump(mode="json")
        result_dict["scenario_description"] = f"live_{asset}_{ts.date()}"
        _save_debate_result(result_dict)

        import telegram_service
        asyncio.create_task(telegram_service.notify_debate_complete(
            asset=asset, action=fm.final_action, conviction=fm.conviction,
            regime_label=context.regime_label, rationale=fm.rationale,
        ))

        await websocket.send_json({"type": "round1_start"})
        for thesis in theses:
            t = thesis if isinstance(thesis, dict) else thesis.model_dump()
            await websocket.send_json(_thesis_to_msg(t))
            await asyncio.sleep(0.2)

        await websocket.send_json({
            "type": "risk_limits",
            "max_position_size": risk.max_position_size,
            "volatility_forecast": risk.volatility_forecast,
            "cvar_95": risk.cvar_95,
            "risk_regime": risk.risk_regime,
        })

        if rebuttals:
            await websocket.send_json({"type": "round2_start"})
            for r in rebuttals:
                rb = r if isinstance(r, dict) else r.model_dump()
                await websocket.send_json(_rebuttal_to_msg(rb))
                await asyncio.sleep(0.1)

        await websocket.send_json({
            "type": "fund_manager",
            "action": fm.final_action,
            "position_size": fm.position_size,
            "conviction": fm.conviction,
            "rationale": fm.rationale,
            "regime_override": fm.regime_override,
            "dissenting_agents": fm.dissenting_agents,
        })
        await websocket.send_json({"type": "ppo_decision", "action": fm.final_action, "position_size": fm.position_size, "regime_override": fm.regime_override})
        await websocket.send_json({"type": "debate_complete"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("stream_debate_live failed")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


def get_debates_list() -> list[dict]:
    debates = _load_debates()
    return [
        {
            "id": i,
            "timestamp": d.get("timestamp"),
            "asset": d.get("asset"),
            "regime_label": d.get("regime_label"),
            "scenario": d.get("scenario_description", ""),
            "action": d.get("fund_manager_decision", {}).get("final_action", "hold"),
            "conviction": d.get("fund_manager_decision", {}).get("conviction", 0),
        }
        for i, d in enumerate(debates)
    ]


def get_debate_detail(debate_id: int) -> dict | None:
    """Full stored debate — theses, rebuttals, risk limits, decision, tokens, duration."""
    debates = _load_debates()
    if debate_id < 0 or debate_id >= len(debates):
        return None
    return {"id": debate_id, **debates[debate_id]}
