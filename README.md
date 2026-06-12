# Multi-Instrument Paper Trading Bot

A paper-trading bot that watches five instruments around the clock, runs a
different strategy on each market's natural timeframe, keeps risk constant
with ATR-based sizing and a hard 1% stop, and sends you two Telegram
messages a day. Execution goes to a **free Alpaca paper-trading account**
(real broker infrastructure, simulated money); without Alpaca keys it
falls back to a fully internal simulator on free Yahoo Finance data.

**This trades paper money only.** It's for validating the strategies, not
for getting rich this week.

## What it trades

Alpaca doesn't carry futures, so the indices and commodities trade as
their most liquid ETF proxies. Bitcoin is real spot crypto.

| Instrument | Ticker | Timeframe | Strategy | Why |
|---|---|---|---|---|
| S&P 500 | `SPY` | 15m | Mean reversion | Indices overextend a little every few hours, then snap back |
| Nasdaq 100 | `QQQ` | 15m | Mean reversion | Same, with more juice |
| Bitcoin | `BTC/USD` | 1h | Momentum breakout (long-only on Alpaca — spot crypto can't be shorted) | Crypto trends harder — ride the move, don't fade it |
| Gold | `GLD` | 4h | Trend following | Commodities move in cleaner waves; skip the intraday noise |
| Oil (WTI) | `USO` | 4h | Trend following | Same |

## Risk rules

- **ATR sizing** — every position is sized so the loss at its stop is 1%
  of account equity. A quiet day on gold gets a bigger size than a
  volatile day on bitcoin; dollar risk stays constant.
- **Notional cap** — position notional is capped at 100% of equity
  (`max_position_notional_pct`), so tight intraday stops can't imply
  leveraged orders; when the cap bites, the trade simply risks less
  than the full 1%.
- **Hard stop, no exceptions** — the stop is fixed at entry (2 × ATR from
  the actual fill), checked at every candle close *and* swept against the
  latest trade price every minute in between.
- **Correlation filter** — S&P, Nasdaq and Bitcoin are one risk-on bucket.
  At most two of them can be open in the same direction; the third signal
  is skipped and logged.

## Daily messages

- **7am** — market briefing: price and 24h change per instrument,
  volatility regime, what each strategy is watching, open positions.
- **9pm** — performance report: trades closed today with P&L, open
  positions, equity, all-time win rate, and any signals the risk rules
  skipped.

Times and timezone are set in `config.yaml` (default `America/New_York`).

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/professionaltraderkit/Tradingbot.git
cd Tradingbot
pip install -r requirements.txt
cp .env.example .env
```

### Alpaca paper account (free)

1. Sign up at [alpaca.markets](https://app.alpaca.markets/signup) — no
   funding needed for paper trading.
2. In the dashboard, switch to the **Paper** account and generate API keys.
3. Put them in `.env` as `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`.

This gives real-time IEX market data, real order simulation against live
quotes, and a one-line switch to live trading later (`paper=True` in
`bot/broker.py` — don't, until the paper results earn it).

No keys? The bot logs a warning and runs on the internal simulator with
Yahoo Finance data instead (`broker.mode: simulated` forces this).

### Telegram (optional but recommended)

1. Message [@BotFather](https://t.me/BotFather) on Telegram, send
   `/newbot`, follow the prompts, and copy the **bot token**.
2. Message your new bot anything (e.g. "hi") so it can reply to you.
3. Get your **chat id**: open
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser and
   read `chat.id` from the response.
4. Fill `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`.

Without a token the bot still runs — reports print to the console instead.

## Running

```bash
python run.py                  # the live loop — leave it running
python run.py --once           # single evaluation pass, then exit
python run.py --report morning # send the morning briefing right now
python run.py --report evening # send the evening report right now
```

The bot evaluates each instrument when its candle closes (15m / 1h / 4h),
enforces stops first, sweeps stops every minute between candles, then acts
on signals. Local state lives in `tradingbot.db` (SQLite) — stop and
restart the bot without losing track of open positions; on Alpaca the
account itself is the source of truth for equity and fills.

Keep it alive on a laptop with something like:

```bash
nohup python run.py >> bot.log 2>&1 &
```

## The second bot: Level Reversal

`config-levels.yaml` defines a separate bot instance that day-trades SPY
and QQQ on 5-minute candles by fading key intraday levels: the previous
session's high/low and today's premarket high/low act as resistance and
support. When a bar pierces a level and gets rejected (closes back on the
original side with a counter-directional body), it shorts the failed
breakout or buys the rejection, takes profit into the next level, and is
always flat by 15:55 ET — no overnight risk.

Run it alongside (or instead of) the main bot — each has its own config,
database, history and reports:

```bash
python run.py --config config-levels.yaml             # the live loop
python run.py --config config-levels.yaml --status    # its positions/trades
python backtest.py --config config-levels.yaml        # backtest it
```

Both bots share the same Alpaca paper account and Telegram channel
(reports are labeled with each bot's name), so don't put the same ticker
in both configs at once.

## Backtesting

```bash
python backtest.py             # full available history per instrument
python backtest.py --days 30   # restrict to the last 30 days
```

Runs the exact same strategy, sizing and stop code over historical candles
(Alpaca data if keys are set, Yahoo otherwise) and prints per-instrument
stats. Backtests always use the internal simulator — no orders are sent.

## Tuning

Everything lives in `config.yaml`: instruments, strategy parameters,
risk-per-trade, ATR stop multiple, notional cap, the correlation group and
cap, report times, starting equity, broker mode. No code edits needed for
parameter changes.

## Tests

```bash
python -m pytest
```

Covers the indicator math, the 1%-risk sizing and notional cap, the
correlation filter, stop enforcement and P&L accounting, Alpaca order
mechanics against a mocked broker (fill slippage, quantity rounding,
stop anchoring), plus an end-to-end engine test on synthetic data
(no network needed).

## Project layout

```
run.py                 entrypoint: live loop / --once / --report
backtest.py            historical simulation using the same components
config.yaml            all tunables
bot/data.py            Alpaca market data (IEX stocks + crypto), Yahoo fallback
bot/broker.py          Alpaca trading API wrapper (paper by default)
bot/indicators.py      SMA, EMA, RSI, ATR, z-score, Donchian channels
bot/strategies/        mean_reversion, momentum_breakout, trend_following
bot/risk.py            ATR sizing, notional cap, stops, correlation filter
bot/portfolio.py       PaperPortfolio (simulator) + LivePortfolio (Alpaca)
bot/state.py           SQLite persistence (positions, trades, events)
bot/engine.py          scheduling, stop sweep, signal→risk→fill pipeline
bot/reporter.py        7am briefing and 9pm report content
bot/notify.py          Telegram delivery (console fallback)
tests/                 unit + integration tests
```

## Disclaimer

This is an educational paper-trading project. Nothing here is financial
advice, and past (simulated) performance says little about the future. If
you ever adapt it to trade real money, that's on you.
