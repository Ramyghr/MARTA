# Kaggle Debate Coverage Run — LLM Debate-Layer Audit Prerequisite

Generates dense, real-LLM debate coverage (BTC_USDT, daily cadence, all 6 regimes, full
2024 validation window — 366 candidate timestamps) so the LLM debate layer's actual
trading-performance contribution can be measured, instead of relying on the current
~1%-of-2024-hours debate cache. This is the blocking dependency for that audit
(`docs/RESULTS_AND_METHODOLOGY.md` Section 3/6) — not the audit itself.

Runs on Kaggle, not locally, so it doesn't need your PC open — see Step 4's "Save & Run
All (Commit)" note, which runs server-side even after you close the tab.

Local package already prepared and verified this session: `kaggle_upload/marta-debate-coverage-pkg/`
(91 MB — code + the 5 crypto5 aligned parquets + causal regime stream + regime semantics
+ news cache; **no FinBERT weights**). Data load + timestamp selection already
smoke-tested against this exact package: 366 picks, all 6 regime labels covered,
window 2024-01-01 → 2024-12-31.

**Correction (found running this for real):** `agents/sentiment_agent.py` does **not**
fall back to pulling `ProsusAI/finbert` from Hugging Face on its own — it only ever
checks a local path (`{MARTA_ROOT}/models/finbert`), and silently returns a neutral
thesis on every debate if that path doesn't exist. The earlier claim in this doc that it
"pulls from Hugging Face instead" was aspirational, not what the code does. Step 2b
below downloads the weights into that exact path explicitly — do not skip it, or
SentimentAgent is dead weight across all 366 debates with no error to tell you so.

---

## Step 0 — create the Kaggle dataset (one-time)

Kaggle → Datasets → New Dataset → upload the contents of
`kaggle_upload/marta-debate-coverage-pkg/` → name it `marta-debate-coverage-pkg`.

(Or via CLI, if you have the `kaggle` tool configured:
`kaggle datasets create -p kaggle_upload/marta-debate-coverage-pkg -m "debate coverage pkg"`.)

---

## Step 1 — new Kaggle Notebook

Settings (right sidebar):
- **Accelerator: GPU T4 x2** (or P100) — this is what gets Mistral calls down to the
  ~30-70s/debate documented in CLAUDE.md, vs 300s+ on CPU. At 366 debates, CPU-only
  would likely not finish in one session.
- **Internet: On** — required for the Ollama install script, the `mistral:7b-instruct`
  pull (~4.1 GB), and the FinBERT download from Hugging Face.
- **Add Data** → attach the `marta-debate-coverage-pkg` dataset from Step 0.

---

## Step 2 — Notebook Cell 1: stage the package into a writable directory

Kaggle mounts datasets read-only under `/kaggle/input/`; the script needs to write
`data/debates/debate_results.json`, so copy everything into `/kaggle/working/` first.

```python
!cp -r /kaggle/input/marta-debate-coverage-pkg /kaggle/working/marta
%cd /kaggle/working/marta
!mkdir -p data/debates
```

## Step 3 — Notebook Cell 2: install Python deps

```python
!pip install -q langchain-ollama langchain-core langgraph pydantic rich transformers arch pyarrow
# pandas / numpy / scipy / torch are already on the Kaggle GPU image.
```

## Step 3b — Notebook Cell 2b: download FinBERT weights

Required — see the correction above. Without this, `SentimentAgent` silently returns
"neutral" for every single debate, no error raised.

```python
from huggingface_hub import snapshot_download
snapshot_download("ProsusAI/finbert", local_dir="/kaggle/working/marta/models/finbert")
```

## Step 4 — Notebook Cell 3: install + start Ollama, pull the model

The stock Ollama install script needs `zstd` to unpack its archive, which is not on the
Kaggle base image (`ERROR: This version requires zstd for extraction` /
`FileNotFoundError: [Errno 2] No such file or directory: 'ollama'` if skipped) —
install it first.

```python
!apt-get -qq update && apt-get -qq install -y zstd
!curl -fsSL https://ollama.com/install.sh | sh

import subprocess, time, requests

ollama_proc = subprocess.Popen(
    ["ollama", "serve"],
    stdout=open("/kaggle/working/ollama.log", "w"),
    stderr=subprocess.STDOUT,
)

for _ in range(30):
    try:
        requests.get("http://localhost:11434")
        print("Ollama is up")
        break
    except Exception:
        time.sleep(2)
else:
    raise RuntimeError("Ollama didn't come up — check /kaggle/working/ollama.log")

!ollama pull mistral:7b-instruct
```

## Step 5 — Notebook Cell 4: sanity check before the real run

Mirrors this project's own established Kaggle discipline (see
`kaggle_retrain_instructions_v3.md` Step 3) — verify data and a single live debate
before committing to a multi-hour run.

```python
import sys
sys.path.insert(0, "/kaggle/working/marta")

from utils.freshness import CRYPTO5_DEBATE_ASSETS, CRYPTO5_PROCESSED, load_debate_data
regime_stream, aligned_dfs = load_debate_data(processed_dir=CRYPTO5_PROCESSED, assets=CRYPTO5_DEBATE_ASSETS)
assert len(aligned_dfs) == 5, f"expected 5 assets, got {list(aligned_dfs.keys())}"
print("Data OK:", len(regime_stream), "regime rows,", list(aligned_dfs.keys()))

# One real debate, timed — confirms the full agent chain + Ollama + FinBERT work
# end-to-end before the long run, and gives a real per-debate time estimate.
import time
sys.path.insert(0, "/kaggle/working/marta/scripts")
from kaggle_debate_coverage import _select_timestamps
from agents.macro_agent import MacroAgent
from agents.technical_agent import TechnicalAgent
from agents.risk_agent import RiskAgent
from agents.fund_manager import FundManagerAgent
from agents.sentiment_agent import SentimentAgent
from agents.contrarian_agent import ContrarianAgent
from agents.graph import DebateGraph, run_debate
from functools import partial
from utils.freshness import CRYPTO5_SEMANTICS_PATH, CRYPTO5_PROCESSED as _DIR, CRYPTO5_DEBATE_ASSETS as _ASSETS

fm_agent = FundManagerAgent(semantics_path=CRYPTO5_SEMANTICS_PATH)
graph = DebateGraph(
    MacroAgent(assets=_ASSETS, data_dir=_DIR), TechnicalAgent(), RiskAgent(data_dir=_DIR),
    fm_agent, SentimentAgent(macro_assets={"macro"}, fear_greed_assets=set(_ASSETS)),
    ContrarianAgent(regime_stream_df=regime_stream),
)
picks = _select_timestamps(regime_stream, "BTC_USDT", 24, True)
t0 = time.time()
result = partial(run_debate, semantics_path=CRYPTO5_SEMANTICS_PATH)(
    picks[0][0], "BTC_USDT", regime_stream, aligned_dfs, graph
)
elapsed = time.time() - t0
print(f"Single debate OK in {elapsed:.1f}s -> {result.fund_manager_decision.final_action} "
      f"(conviction={result.fund_manager_decision.conviction:.3f})")
print(f"Estimated total for {len(picks)} debates: {elapsed * len(picks) / 3600:.1f}h")
```

If `elapsed` comes back over ~90s, GPU acceleration likely isn't active — check the
Accelerator setting before running the full sweep (CPU-only won't finish in one session).

## Step 6 — Notebook Cell 5: the real run

```python
!cd /kaggle/working/marta && python scripts/kaggle_debate_coverage.py --stride-hours 24 --all-regimes --asset BTC_USDT
```

**Run this via "Save & Run All (Commit)"** (top-right of the notebook), not by clicking
through cells interactively — commit runs execute server-side on Kaggle's infrastructure
and keep going even after you close the browser tab or shut down your PC, which is the
whole point here. You can check back on progress from the notebook's "Viewer"/versions
page.

The script writes `data/debates/debate_results.json` after every single debate (atomic
temp-file + rename), so if the session times out or the GPU quota runs out partway
through, nothing already completed is lost — re-running the same commit later will skip
every timestamp already in the file and pick up where it left off.

## Step 7 — after the run: bring the result back

Kaggle → your notebook → Output tab → download `data/debates/debate_results.json`.

Locally, merge it into the existing cache (append, dedup by timestamp — same pattern
`run_weekly_debates.py` already uses):

```python
import json

existing = json.load(open("data/debates/debate_results.json"))
existing_ts = {r["timestamp"] for r in existing}

new = json.load(open("~/Downloads/debate_results.json"))  # from Kaggle
merged = existing + [r for r in new if r["timestamp"] not in existing_ts]

json.dump(merged, open("data/debates/debate_results.json", "w"), indent=2, default=str)
print(f"{len(existing)} existing + {len(merged) - len(existing)} new = {len(merged)} total")
```

That gives you the dense BTC coverage the debate-layer audit needs. The audit itself
(rule-based sizing on this consensus vs. buy-hold/technical-rule baselines, block-
bootstrap significance) is separate follow-up work, not part of this run.
