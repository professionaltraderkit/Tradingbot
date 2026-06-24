# PEAD Toolkit — Post-Earnings Announcement Drift

A long-only swing strategy that buys stocks right after they post a big earnings
**beat** and holds them for ~60 trading days, harvesting the well-documented
"drift" as the market slowly re-prices the surprise.

Validated on 5,961 earnings events (2005–2026, 71 liquid large caps):
**~8–9%/yr market-excess alpha, Sharpe ~0.9, no decay into 2019–2026.**

---

## 1. The idea in plain English

When a company reports earnings that beat expectations, the stock jumps on the
day — but it **doesn't jump enough**. Investors under-react, and the price keeps
grinding higher for weeks. Symmetrically, big misses keep drifting down. This is
the oldest, most-replicated anomaly in finance (Ball & Brown 1968; Bernard &
Thomas 1989).

**What our testing found for a modern retail trader:**

| Finding | Implication |
|---|---|
| Drift only shows up over **20–60 days** | It's a *swing* edge, not a day-trade. The first 1–5 days are noise. |
| The **short leg is dead** (big misses still drift up in large caps) | Trade it **long-only**. Don't short the losers. |
| Effect is **strongest in the top quintile** of surprises | Only buy the *biggest* beats, not every beat. |
| **SUE + EAR combined** = sweet spot (~12.9%/yr, 58% win) | Best beats that *also* rose on the news drift hardest. |
| **No decay** across 2005–11 / 2012–18 / 2019–26 | Still works today (unlike pairs trading, which died). |

### The two signals

- **SUE** — Standardized Unexpected Earnings. We use the reported
  `Surprise(%) = (actual EPS − estimate) / |estimate|`. The accounting surprise.
- **EAR** — Earnings Announcement Return. The stock's 3-day return *around* the
  report, minus SPY. The *price reaction* itself. When SUE and EAR agree (big
  beat AND the stock popped), the drift is most reliable.

### The trading rule

> **Long-only. Buy the top-quintile earnings beats at the close the day after
> the report. Equal-weight. Hold ~60 trading days. Exit on schedule.**
> Optionally also require a positive EAR ("combined") for higher conviction.

---

## 2. The five scripts

All in `C:\Users\xxdis\Tradingbot\tradingview\`. Python 3, needs
`yfinance pandas numpy statsmodels scipy` (all installed).

| Script | What it does | When you run it |
|---|---|---|
| **`pead_backtest.py`** | Pulls earnings + prices for the universe, builds the event table, prints quintile drift by horizon. Writes `pead_events.csv`. | Once, to (re)generate the dataset. |
| **`pead_analyze.py`** | Reads the CSV (no download). t-stats, long-only sizing, combined SUE+EAR, decay-by-era. | To re-examine significance without re-downloading. |
| **`pead_portfolio.py`** | Turns events into a real held portfolio → Sharpe, CAGR, max drawdown vs SPY. | To judge tradeability / risk. |
| **`pead_mag7.py`** | Same study restricted to the Magnificent 7 (pulls TSLA fresh). Per-name breakdown. | Mag-7 deep dive. |
| **`pead_live.py`** | **The one you run regularly.** Scans for recent reporters and flags current BUY candidates with exit dates. | Daily/weekly during earnings season. |

---

## 3. How it works under the hood

**Data source:** `yfinance`.
- Earnings surprises: `yf.Ticker(tk).get_earnings_dates(limit=100)` → date +
  `Surprise(%)`. (Yahoo caps `limit` at 100 = ~25 years of quarters.)
- Prices: `yf.download(..., auto_adjust=True)` daily closes, plus SPY as the
  market benchmark.

**Building an event (`pead_backtest.build_events`):** for every past report,
1. find the **first trading day strictly after** the announcement → that close
   is the **entry** (T+1; avoids the earnings-gap look-ahead),
2. compute **EAR** = the 3-day return bracketing the announcement minus SPY,
3. compute **forward returns** at 1/5/20/60 days, each **market-excess** (stock
   return minus SPY over the same window) so we measure *drift, not beta*.

**Quintile sorting:** events are ranked into surprise buckets **within each
calendar quarter**, so a 2008 beat is only compared to other 2008 beats — no
look-ahead, no regime contamination. Long = bucket 5 (biggest beats).

**Portfolio sim (`pead_portfolio.py`):** maintains a rolling book — every
qualifying entry opens an equal-dollar long held 60 days; the portfolio's daily
return is the equal-weight mean of all names currently held (≈13 at a time).
That yields a true equity curve → Sharpe, CAGR, drawdown. Costs charged per side.

**Live screen (`pead_live.py`):** pulls the trailing ~400 days of surprises to
build a percentile distribution, then ranks the last *N* days of reporters
against it. Anything in the **top 20%** (TOP20%) or **top 10%** (TOP10%) of
surprises is a BUY candidate; it prints the suggested exit date (entry + ~60
business days).

---

## 4. How to use it — commands

### First-time / refresh the dataset
```bash
cd /c/Users/xxdis/Tradingbot/tradingview
python pead_backtest.py --start 2005-01-01 --csv pead_events.csv
```
Takes ~2–3 min (downloads ~70 names). Re-run every few months to add new events.

### Inspect the evidence (instant, no download)
```bash
python pead_analyze.py                      # t-stats, long-only, combined, decay
python pead_portfolio.py --hold 60 --bucket quintile   # Sharpe / CAGR / drawdown
python pead_portfolio.py --hold 60 --combined          # SUE+EAR version
python pead_mag7.py                         # Mag-7 per-name breakdown
```

### The live screener — what you run in practice
```bash
python pead_live.py                          # full 71-name universe, last 7 days
python pead_live.py --lookback 21            # widen the window
python pead_live.py --universe mag7          # just the Magnificent 7
python pead_live.py --universe core3          # AMZN / NVDA / AAPL (cleanest)
python pead_live.py --universe core3 --combined   # also require positive EAR
```

**Flags:**
- `--lookback N` — calendar days of reporters to scan (default 7). Use 14–21
  during heavy weeks.
- `--universe full|mag7|core3` — which names to scan.
- `--combined` — only flag BUY if the stock *also* rose on the report (EAR > 0).

---

## 5. Reading the live-screen output

```
ticker   reported  surprise_%  EAR_%   last   rank  exit_~60d BUY
 GOOGL 2026-04-29        94.3    9.0 368.03 TOP10% 2026-07-23 YES
  AMZN 2026-04-29        69.0    1.1 244.39 TOP10% 2026-07-23 YES
  TSLA 2026-04-22        17.1   -3.9 400.49      -  2026-07-16
```

- **surprise_%** — how much it beat (EPS basis).
- **EAR_%** — 3-day market-excess reaction. Positive = market liked it (confirms).
  Negative = "sold the news" (weaker drift; `--combined` filters these out).
- **rank** — TOP10% / TOP20% / `-` vs the trailing year's surprise distribution.
- **exit_~60d** — close the position on/near this date (≈60 trading days out).
- **BUY = YES** — top-quintile beat → an actionable long.

So above: GOOGL and AMZN are buys; TSLA beat but the market sold it (negative
EAR), so it's not flagged.

---

## 6. The trading playbook

1. Run `pead_live.py` during earnings season (peaks **late Jan / Apr / Jul / Oct**).
   Between seasons it's correctly empty.
2. For each **BUY = YES**, go long at/near the next close, equal-weight across
   all current signals (the backtest held ~13 at once; don't put it all in one).
3. **Hold ~60 trading days** — exit on the printed `exit_~60d` date. Don't bail
   early on the first wobble; the drift is a slow grind, win rate is only ~52–58%.
4. Optional risk overlay: a hard stop (e.g. −12 to −15%) caps the occasional
   blow-up; the raw study has no stop.
5. **Mag-7 nuance:** favor **AMZN / NVDA / AAPL** (clean drifters). MSFT and META
   barely drift — skip. TSLA drifts hardest but beats only ~58% of the time.

---

## 7. Honest caveats

- **Survivorship bias** — the universe is *currently-listed* names, which inflates
  absolute returns somewhat. The persistence and t-stats are still robust.
- **Long-only** — the short leg doesn't pay in large caps (quality names recover).
- **Large-cap only** — PEAD is documented *stronger* in small/mid caps (more
  under-reaction), so this likely **understates** the available edge — but small
  caps cost more to trade. The `--universe full` list is the liquid compromise.
- **Beta is in the absolute return** — the −51% portfolio drawdown is just being
  long equities through 2008/2020. The strategy's *own* risk (the market-excess
  alpha curve) drew down only ~−22%.
- **Not a TradingView/Pine strategy** — Pine can't fetch analyst estimates, so
  this lives as a Python screener, not an on-chart backtest.

---

*Built 2026-06-20. See also the memory note `pead-strategy.md` for the full
result tables.*
