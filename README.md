# Multi-Instrument Paper Trading Bot

A paper-trading bot that watches five instruments around the clock, runs a
different strategy on each market's natural timeframe, keeps risk constant
with ATR-based sizing and a hard 1% stop, and sends you two Telegram
messages a day. No broker account, no API costs — market data comes from
Yahoo Finance and all fills are simulated.

**This trades paper money only.** It's for validating the strategies, not
for getting rich this week.

## What it trades

| Instrument | Ticker | Timeframe | Strategy | Why |
|---|---|---|---|---|
| S&P 500 futures | `ES=F` | 15m | Mean reversion | Indices overextend a little every few hours, then snap back |
| Nasdaq 100 futures | `NQ=F` | 15m | Mean reversion | Same, with more juice |
| Bitcoin | `BTC-USD` | 1h | Momentum breakout | Crypto trends harder — ride the move, don't fade it |
| Gold futures | `GC=F` | 4h | Trend following | Commodities move in cleaner waves; skip the intraday noise |
| WTI crude futures | `CL=F` | 4h | Trend following | Same |

## Risk rules

- **ATR sizing** — every position is sized so the loss at its stop is
  exactly 1% of account equity. A quiet day on gold gets a bigger size
  than a volatile day on bitcoin; dollar risk stays constant.
- **Hard stop, no exceptions** — the stop price is fixed at entry
  (2 × ATR away) and checked before anything else on every cycle.
- **Correlation filter** — S&P, Nasdaq and Bitcoin are one risk-on bucket.
  At most two of them can be open in the same direction; the third signal
  is skipped and logged.

## Daily messages

- **7am** — market briefing: price and 24h change per instrument,
  volatility regime, what each strategy is watching, open positions.
- **9pm** — performance report: trades closed today with P&L, open
  positions, equity, all-time win rate, and any signals the correlation
  filter blocked.

Times and timezone are set in `config.yaml` (default `America/New_York`).

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/professionaltraderkit/Tradingbot.git
cd Tradingbot
pip install -r requirements.txt
```

### Telegram (optional but recommended)

1. Message [@BotFather](https://t.me/BotFather) on Telegram, send
   `/newbot`, follow the prompts, and copy the **bot token**.
2. Message your new bot anything (e.g. "hi") so it can reply to you.
3. Get your **chat id**: open
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser and
   read `chat.id` from the response.
4. Configure the bot:

```bash
cp .env.example .env
# edit .env and fill in TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
```

Without a token the bot still runs — reports print to the console instead.

## Running

```bash
python run.py                  # the live loop — leave it running
python run.py --once           # single evaluation pass, then exit
python run.py --report morning # send the morning briefing right now
python run.py --report evening # send the evening report right now
```

The bot evaluates each instrument when its candle closes (15m / 1h / 4h),
enforces stops first, then acts on signals. State lives in `tradingbot.db`
(SQLite) — you can stop and restart the bot without losing open positions.

Keep it alive on a laptop with something like:

```bash
nohup python run.py >> bot.log 2>&1 &
```

## Backtesting

```bash
python backtest.py             # full available history per instrument
python backtest.py --days 30   # restrict to the last 30 days
```

Runs the exact same strategy, sizing and stop code over historical candles
and prints per-instrument stats (trades, win rate, P&L, max drawdown).
Yahoo limits intraday history: ~60 days of 15m data, ~2 years of 1h/4h.

## Tuning

Everything lives in `config.yaml`: instruments, strategy parameters,
risk-per-trade, ATR stop multiple, the correlation group and cap, report
times, starting equity. No code edits needed for parameter changes.

## Tests

```bash
python -m pytest
```

Covers the indicator math, the 1%-risk sizing, the correlation filter,
stop enforcement and P&L accounting, plus an end-to-end engine test on
synthetic data (no network needed).

## Project layout

```
run.py                 entrypoint: live loop / --once / --report
backtest.py            historical simulation using the same components
config.yaml            all tunables
bot/data.py            Yahoo Finance fetching, 1h→4h resampling
bot/indicators.py      SMA, EMA, RSI, ATR, z-score, Donchian channels
bot/strategies/        mean_reversion, momentum_breakout, trend_following
bot/risk.py            ATR sizing, hard stop calc, correlation filter
bot/portfolio.py       paper broker: fills, stops, P&L
bot/state.py           SQLite persistence (positions, trades, events)
bot/engine.py          scheduling + signal→risk→fill pipeline
bot/reporter.py        7am briefing and 9pm report content
bot/notify.py          Telegram delivery (console fallback)
tests/                 unit + integration tests
```

## Disclaimer

This is an educational paper-trading project. Nothing here is financial
advice, and past (simulated) performance says little about the future. If
you ever adapt it to trade real money, that's on you.
