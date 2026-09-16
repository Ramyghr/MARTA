# Known Bugs — triage + fix instructions

Written 2026-09-14 under time pressure (session budget nearly exhausted) — documentation
only, nothing below has been fixed yet. Each entry has enough detail to fix without
re-investigating from scratch. Ordered by recommended fix order, not severity.

## 0. [ROOT-CAUSED 2026-09-15 — was transient, defensive fix applied anyway] DNS/network noise

Everything pasted from the `backend-1` docker logs — Alpaca, Binance `exchangeInfo`,
`testnet.binance.vision`, NewsAPI, StockTwits, 8 different RSS feeds, Fear & Greed — failed
with the identical error shape: `NameResolutionError(...Failed to resolve '<host>')`.

**Checked directly against your actual running `dashboard-backend-1` container:**
`docker exec dashboard-backend-1 getent hosts newsapi.org` resolved fine, and
`docker logs dashboard-backend-1 --since 10m` shows zero DNS errors right now. **The outage
had already cleared by the time this was checked** — it was a transient network blip (the
host's own DNS looked fine too), not a persistent container misconfiguration. Bugs #2/#4's
"tied to bug #0" theories in earlier revisions of this doc were consequently overstated —
see their updated entries below.

**Root cause identified anyway, and defended against for next time:** the container's
`/etc/resolv.conf` was correctly auto-configured by Docker (`nameserver 127.0.0.11` →
`8.8.8.8`/`8.8.4.4` upstream), so this wasn't a config bug — just bad luck that the upstream
was briefly unreachable. Pinned explicit DNS servers on the backend service in
`dashboard/docker-compose.yml` (`dns: [8.8.8.8, 1.1.1.1]`) anyway, as a defensive hardening
measure so a future blip is less likely to cascade the same way. Not proven to prevent a
recurrence (the original failure mode wasn't a resolv.conf misconfiguration), just a
reasonable belt-and-suspenders change.

**If this happens again:** `docker exec dashboard-backend-1 getent hosts <host>` is the
fastest direct check against the real container (no need to guess from log noise).

---

## 1. [FIXED 2026-09-14] Bootstrap CI key mismatch on the backtest page

**Symptom:** `/backtest` page's "Ablation Study" significance callouts throw / render
wrong — confirmed via `pytest tests/test_api_contracts.py::test_rl_results_ci_key_names`,
currently FAILING (this test exists specifically as an "F-19 regression" guard, meaning
this was fixed once before and has regressed).

**Root cause:** `dashboard/backend/rl_service.py:73-74` outputs each `bootstrap_ci` entry
as `{"ci_low": ..., "ci_high": ...}`. Two frontend files disagree about the contract:
- `dashboard/frontend/src/components/BacktestPanel.tsx:130` and
  `dashboard/frontend/src/lib/types.ts:176-177` correctly use `ci_low`/`ci_high` — matches
  what the backend actually sends.
- `dashboard/frontend/src/lib/api.ts:23` types `bootstrap_ci` as
  `{ ci_lower_95, ci_upper_95 }`, and `dashboard/frontend/src/app/backtest/page.tsx` (lines
  119, 124, 720, 723) reads `ci.ci_lower_95.toFixed(2)` directly off that data — which is
  `undefined` given what the backend sends, so `.toFixed()` throws a `TypeError` at render
  time on that page.

The regression test asserts the **`ci_lower_95`/`ci_upper_95` naming is the intended
contract** (and that `ci_low`/`ci_high` must NOT be present), so the fix is to make the
backend match the test/`backtest/page.tsx`/`api.ts`, not the other way around.

**Fix (3 files, ~6 lines, no logic changes):**
1. `dashboard/backend/rl_service.py:73-74` — rename output keys:
   ```python
   "ci_lower_95": round(float(v.get("ci_lower_95", 0)), 3),
   "ci_upper_95": round(float(v.get("ci_upper_95", 0)), 3),
   ```
2. `dashboard/frontend/src/components/BacktestPanel.tsx:130` — `ci.ci_low`/`ci.ci_high` →
   `ci.ci_lower_95`/`ci.ci_upper_95`.
3. `dashboard/frontend/src/lib/types.ts:176-177` — rename the two fields to
   `ci_lower_95: number` / `ci_upper_95: number`.
4. Run `pytest tests/test_api_contracts.py -k ci_key_names` to confirm green, then
   `docker compose up -d --force-recreate backend frontend` and reload `/backtest`.

**Status: fixed.** Renamed `rl_service.py`'s output keys to `ci_lower_95`/`ci_upper_95`,
updated `BacktestPanel.tsx` and `types.ts` to match. Verified: `pytest
tests/test_api_contracts.py -k ci_key_names` passes; grepped the whole frontend + backend
for `ci_low`/`ci_high` afterward, zero stray references left. Not yet checked in an actual
browser (budget) — worth a quick visual pass on `/backtest` next time the dashboard is up.

---

## 2. [NOT A REAL BUG — test-mock gap, fixed 2026-09-14] `/api/regime/BTC_USDT` returns an empty body

**Symptom:** `pytest tests/test_api_contracts.py::test_get_regime_has_label` FAILS —
`GET /api/regime/BTC_USDT` returns `{}` (no `regime_label` key) when queried at "now."

**Root cause (traced, not yet fixed):** `regime_service.get_live_regime_crypto5()`
(`dashboard/backend/regime_service.py:152-200`) falls back to the static historical lookup
(`get_regime()`) at three points if anything is off — model not loaded (line 166-167), live
parquet file missing (line 170-171), or live parquet has fewer than `_LIVE_LOOKBACK_HOURS`
rows (line 176-177) — and the static fallback is itself apparently returning nothing usable
for "now" (2026-09-14), producing `{"live": False}` with no `regime_label`.

**Likely tied to bug #0** — the crypto5 live bridge (`{asset}_normalized_live.parquet`) is
fed by live-fetched Binance bars; the pasted logs show `[historical] ETH_USDT: parquet ends
2026-07-12 (1529h ago) — fetching extension` immediately followed by `Binance extension
fetch failed ... exchangeInfo` and `no extension data returned` — i.e. the live bridge
can't update *because* of the DNS outage, so it's stale/short, so it can't satisfy the
168h lookback window, so it falls back to a static file that's also stale.

**Do this, in order:**
1. Fix bug #0 (network/DNS) first.
2. Re-run `pytest tests/test_api_contracts.py::test_get_regime_has_label`. If it now
   passes, this was entirely a downstream symptom — done, no code change needed.
3. If it still fails: add a print/log right before each of the three fallback returns in
   `get_live_regime_crypto5()` (lines 167, 171, 177) so you can see *which* branch is
   firing, then check whether `_crypto5_hmm_model` loaded at startup (grep backend startup
   logs for "regime_service") and whether
   `data/processed/crypto5_multihorizon/BTC_USDT_normalized_live.parquet` exists and how
   many rows/how recent it is (`pd.read_parquet(path).tail()`).

**Status: resolved, and it was good news.** The theory above (bug #0 starving the live
bridge) was wrong — checked directly: `data/processed/crypto5_multihorizon/
BTC_USDT_normalized_live.parquet` has 72,822 rows through **2026-09-14 15:00**, fully
fresh, well past the 168h lookback requirement. Called `regime_service
.get_live_regime_crypto5("BTC_USDT", now)` directly (bypassing the API) and it returned a
real, correct result (`regime_label: 2`, "Low Volatility / Range-bound (ETH-dominant)",
proper probability vector) — the actual regime computation works fine.

The pytest failure was a **test-mock gap, not a production bug**:
`tests/test_api_contracts.py` mocks the entire `regime_service` module and only configured
a return value for `.get_regime` — but `/api/regime/{asset}` (main.py:225-230) calls
`.get_live_regime`, a different function. The unconfigured mock silently returned a bare
`MagicMock`, which serializes to `{}`, regardless of what the real backend does. Fixed by
adding `_MOCK_REGIME_SERVICE.get_live_regime.return_value = {...}` with a realistic shape.
`pytest tests/` now 142/143 (only the pre-existing stale-`EURUSD` test left, unrelated).

**Take the DNS/network issue (#0) less seriously as an explanation for missing regime
data** than this doc originally implied — the crypto5 live bridge for BTC_USDT is
confirmed healthy right now. If the *live dashboard* still shows a blank regime for
BTC_USDT, that's a genuinely new symptom worth re-reporting with specifics (which page,
what exactly renders), not a re-occurrence of this one.

---

## 3. [STILL OPEN — bug #0 ruled out as the explanation] High latency across the UI

Since bug #0 turned out to be a transient blip that had already cleared, it's no longer a
credible explanation for latency you're still observing right now. Two remaining
candidates, cheapest-to-check first:

1. **Re-measure fresh.** CLAUDE.md documents a *previous* latency fix
   (`live_signal.py`'s `_fetch_latest_price()` rebuilding a `ccxt.binance()` client per
   call, fixed by caching it + restricting to spot markets, brought
   `/api/rl/live-signal` from ~4.8s to ~0.55s). If that endpoint specifically is slow again,
   check whether something re-introduced a per-call client rebuild.
2. **Already-diagnosed-but-not-fixed:** CLAUDE.md's dashboard audit notes "no shared
   fetch/cache layer (no React Query/SWR, just per-component `useEffect` + raw `fetch`)" —
   N mounted components independently fetch the same conceptual data, so N round-trips is
   inherent to the current architecture. This is an architecture change (introduce a shared
   query/cache layer), not a one-line fix — budget it separately if it's the actual cause.

**How to diagnose:** open the browser Network tab on `/dashboard`, sort by duration, and
see whether slow requests are (a) hitting external hosts, (b) genuine backend compute (GARCH
refit, HMM live-regime recompute), or (c) many small parallel requests with no caching.
Each has a different fix. Not measured live this session — needs your own observation of
which specific requests are slow, since "high latency" alone doesn't point at one cause.

---

## 4. [FIXED 2026-09-15] Chart shows only ~1 week, missing today

**"Missing today" was bug #0** (transient DNS blip breaking the live-extension fetch) — no
longer reproducing now that network is confirmed healthy.

**"Only shows ~1 week" was real, independent of #0, and is now fixed.** Root cause:
`historical.get_history(asset, n=200)` returns the *last 200 rows* — at hourly bars that's
~8.3 days, matching the "1 week" complaint almost exactly. Three call sites used this
undersized default: `/api/history/{asset}` (`main.py`, was `n=200`), and both branches of
the `/ws/price/{asset}` websocket handler (live mode was `limit=300`≈12.5 days, demo/
historical mode was `200`≈8.3 days) — the dashboard's initial chart load goes through the
websocket, not the REST endpoint, so the WS defaults were the ones actually controlling
what a user sees on load.

**Fix:** bumped all three to 720 (30 days). `historical.get_history()` is just a cheap
`.tail(n)` slice, no perf concern; Binance's `fetch_ohlcv` limit param caps at 1000, so 720
is safely within range for the live-fetch path too. Verified directly:
`GET /api/history/BTC_USDT` now returns 720 candles spanning a full 30 days
(`2026-08-16T02:00 → 2026-09-15T01:00` at test time) instead of ~8 days. The user can still
scrub further back via the existing date-range scrubber (`getHistoryRange`, decimates
above 2000 points) — this only changes the *default* window on load.

---

## 5. [DOES NOT REPRODUCE against current source — likely a stale docker image] Switching assets doesn't update the chart until page nav

**Live-repro'd this properly** (`.claude/launch.json`'s `marta-backend-preview`/
`marta-frontend-preview` configs, added this session, on spare ports 8010/3010 so they
don't collide with your running docker stack on 8001/3001 — CORS updated in `main.py` to
allow the new preview origin). Installed a `WebSocket` constructor spy via `preview_eval`
and switched assets three times in a row (ETH → BNB → XRP) with **no page navigation**:

```
construct BNB_USDT
close-called BNB_USDT   (React effect cleanup)
close-called BNB_USDT   (redundant guard at the top of the new effect — harmless)
construct XRP_USDT      (same tick)
closed BNB_USDT         (68ms later — the actual async close handshake completing)
```

Every switch produced a clean close-then-reconnect with no overlap, and the page's own
displayed asset (`"XRP/USDT — historical replay"`) and the `<select>`'s value updated
correctly every time. **Could not reproduce the reported bug against the current source
tree.**

**Most likely explanation: your running docker frontend is stale relative to the source.**
CLAUDE.md's most recent PriceChart rewrite (2026-09-03) specifically added the
`prevAssetRef`/`assetChanged` tracking that's now in the code — plausibly this exact bug
was already fixed then. Per that same CLAUDE.md section: *"the frontend Docker image
bundles its own `node_modules.tar.gz` (not a fresh `npm install`) so that had to be
regenerated and the image rebuilt, not just hot-reloaded."* If the running
`dashboard-frontend-1` container predates that rebuild, it would still show the old,
possibly-broken behavior while the source tree (what I tested against) is already fixed.

**Next step:** `docker compose up -d --force-recreate --build frontend`, then re-test in the
actual browser. If it still reproduces after a real rebuild, that would be a genuinely new
finding worth re-investigating — but test that first, it's much cheaper than more code
archaeology.

---

## 6. [UI GAP FIXED 2026-09-15, underlying scope unchanged by design]

**"Weekly PPO raw only works against BTC"** — the underlying scope is intentional, not a
defect: the native weekly-cadence PPO checkpoint was only ever trained/validated on
`BTC_USDT`. What *was* a real gap: the Strategy Tester dropdown let you select
`weekly_ppo_raw`/`weekly_ppo_raw_regime` for any asset and only found out it was BTC-only
from a 400 after clicking Run.

**Fixed:** `strategy_backtest.list_strategies()` now takes an optional `asset` param and
locks the two weekly-PPO entries whenever `asset != "BTC_USDT"`. `GET
/api/backtest/strategies` accepts `?asset=`, and the frontend
(`backtest/page.tsx`) now re-fetches the strategy list whenever `selectedAsset` changes
(previously fetched once on mount, asset-agnostic) and resets the selected strategy to
`ma_crossover` if it becomes locked out from under the user. Verified directly:
`?asset=BTC_USDT` → both `locked: false`; `?asset=ETH_USDT` → both `locked: true`.
Making it actually work for other assets means training + validating equivalent
weekly-cadence checkpoints for ETH/BNB/LTC/XRP — a research/training task on the order of
hours per asset, out of scope here.

---

## 7. [FIXED 2026-09-14] NewsAPI "Test Connection" in Settings is a stub

**Symptom:** Settings page shows `'newsapi' is not wired to a live integration in this
backend yet.` for NewsAPI's Test Connection button — this is NOT the DNS issue, it's a
different code path.

**Root cause:** `dashboard/backend/main.py:812` — the test-connection endpoint's dispatch
`if/elif` chain has branches for `binance`, `rss`, `twelve_data`, `telegram`, but no
`newsapi` branch, so it falls through to the generic "not wired" message regardless of
whether `NEWSAPI_KEY` is set or the network works. Every other integration's live news
fetch code already exists (`news_live.py` handles NewsAPI elsewhere, per the RSS branch
right above it in the same function) — this is purely a missing dispatch branch in the
Settings test-connection handler, not a missing feature.

**Fix (~10 lines):** add a `newsapi` branch to the dispatch in `main.py` mirroring the
`rss`/`twelve_data` pattern — call whatever function `news_live.py` uses to fetch NewsAPI
headlines directly (bypassing the RSS fallback), and report `ok`/headline count the same
way the `rss` branch does at the top of that function.

**Status: fixed.** Added a `newsapi` branch to `main.py`'s `test_source()` dispatch,
calling the real (already-existing, already-working-elsewhere) `_fetch_newsapi_sync` from
`pipeline/live/news.py` via `asyncio.to_thread`. Hit one real snag: `pipeline` isn't on
`sys.path` by default in this process (same issue `live_feature_bridge.py` already had to
work around) — fixed by inserting `MARTA_ROOT` onto `sys.path` before the import, same
pattern. Verified end-to-end with a manual `TestClient` call: reaches the real function, no
`ModuleNotFoundError`, no crash — returns `{"ok": False, "message": "No articles returned
(bad key, rate limit, or network)"}` in this environment (expected, no working NewsAPI key/
network here). `pytest tests/test_api_contracts.py` still 17/19 (same 2 pre-existing,
unrelated failures as before — see #2 and the stale-EURUSD-test note in the summary below).

**Other Settings-page integrations checked, not stubs — no fix needed:** `binance`, `rss`,
`twelve_data` already had real dispatch branches before this session.  `stocktwits`,
`fear_greed`, and `alpaca` don't go through this dispatch at all — the frontend
(`settings/page.tsx`) uses a `customTest` override for each, hitting their own dedicated
endpoints (`/api/news/stocktwits/{asset}`, `/api/market/fear-greed`, `/api/alpaca/account`)
directly. Confirmed their failures in the pasted logs are the DNS/network issue (#0), not a
missing-wiring problem — `newsapi` was the only source actually falling through to the
generic "not wired" stub. Also confirmed the *host* machine (outside the docker container)
resolves DNS fine (`getent hosts newsapi.org` succeeded) — so #0 is specific to the
backend container's network, not the whole machine; check the container's own DNS/network
setup, not the host's.

---

## Not a bug — pending task to remember

LLM debate notebook has been run for BTC, ETH, BNB, XRP; LTC is running now. Once it
finishes, the resulting JSON needs to be copied back and merged into
`data/debates/debate_results.json` (same pattern as the BTC dense-coverage merge
documented in CLAUDE.md's "Dense BTC debate coverage" section — back up the existing file
first: `cp data/debates/debate_results.json data/debates/debate_results.json.bak_before_ltc_merge`).
Don't forget this once LTC finishes — it's easy to lose track of a background Kaggle run.

---

## Status as of 2026-09-15

| # | Issue | Status |
|---|-------|--------|
| 0 | DNS/network noise | Root-caused (transient blip, already cleared); defensive `dns:` pin added |
| 1 | CI key mismatch | Fixed, test-verified |
| 2 | Regime empty body | Not a real bug — test-mock gap, fixed |
| 3 | High latency | Still open — bug #0 ruled out as the cause, needs fresh live measurement |
| 4 | Chart only ~1 week | Fixed — default window 200/300h → 720h (30 days) |
| 5 | Asset switch doesn't update chart | Does not reproduce against current source (live WS-spy tested); likely your running docker frontend is stale — rebuild and re-test |
| 6 | Weekly PPO BTC-only | UI now greys it out for other assets instead of 400ing; underlying BTC-only scope unchanged by design |
| 7 | NewsAPI stub | Fixed, verified end-to-end |
| — | Stale `EURUSD` test | Fixed alongside the above — `pytest tests/` is 143/143 |

**Only #3 (latency) and #5 (asset-switch, pending a docker rebuild + re-test) remain open.**
Everything else in this document has been fixed and verified this session — via `pytest`,
direct API calls against a live preview backend, and (for #5) a live browser session with
an instrumented WebSocket spy. `.claude/launch.json` gained `marta-backend-preview`/
`marta-frontend-preview` (ports 8010/3010) for future sessions to reuse without touching
the running docker stack on 8001/3001.

**One side effect to know about:** running a second backend instance (for live-testing #5)
briefly conflicted with your live Telegram bot's polling (`Conflict: terminated by other
getUpdates request`) — both instances share the same bot token from `dashboard/backend/
.env`. Harmless and self-resolved once the preview backend was stopped, but worth knowing
if you see that error again after a Claude session touches this project.
