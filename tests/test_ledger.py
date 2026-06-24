"""The experiment ledger: record/query roundtrip on an in-memory database."""

from bot.ledger import ExperimentLedger, params_hash


def _exp(verdict="PASS", oos=4.0, params=None, strategy="vwap_pullback",
         ticker="SPY"):
    return {"strategy": strategy, "ticker": ticker, "name": "S&P 500",
            "timeframe": "5m", "config_file": "config-levels.yaml",
            "data_start": "2026-01-01", "data_end": "2026-03-01", "bars": 4000,
            "cost_bps": 2.0, "n_trials": 30, "n_folds": 3,
            "params": params if params is not None else {"adx_min": 25, "stop_atr": 1.0},
            "is_return": 8.0, "oos_return": oos, "oos_trades": 40, "oos_tstat": 2.3,
            "threshold_t": 1.4, "gap": 0.3, "consistency": 1.0, "verdict": verdict,
            "reasons": [] if verdict == "PASS" else ["OOS return not positive"]}


def test_record_and_recent_roundtrip():
    led = ExperimentLedger(":memory:")
    led.record(_exp(verdict="VETO", oos=-1.0))
    rid = led.record(_exp(verdict="PASS", oos=4.0))
    rows = led.recent(limit=10)
    assert len(rows) == 2
    assert rows[0]["id"] == rid                       # newest first
    assert rows[0]["verdict"] == "PASS"
    assert rows[0]["params"] == {"adx_min": 25, "stop_atr": 1.0}  # JSON decoded
    assert rows[0]["reasons"] == []
    led.close()


def test_best_returns_top_passing_oos():
    led = ExperimentLedger(":memory:")
    led.record(_exp(verdict="PASS", oos=2.0, params={"adx_min": 18}))
    led.record(_exp(verdict="PASS", oos=6.0, params={"adx_min": 25}))
    led.record(_exp(verdict="WEAK", oos=9.0, params={"adx_min": 32}))  # higher, not PASS
    best = led.best("vwap_pullback", "SPY")
    assert best["oos_return"] == 6.0 and best["params"] == {"adx_min": 25}
    led.close()


def test_seen_finds_prior_ruling_by_params():
    led = ExperimentLedger(":memory:")
    params = {"adx_min": 25, "stop_atr": 0.5}
    assert led.seen("vwap_pullback", "SPY", params) is None
    led.record(_exp(verdict="VETO", oos=-2.0, params=params))
    prior = led.seen("vwap_pullback", "SPY", params)
    assert prior is not None and prior["verdict"] == "VETO"
    assert led.seen("vwap_pullback", "SPY", {"adx_min": 99}) is None  # different config
    led.close()


def test_recent_filters_by_strategy_and_ticker():
    led = ExperimentLedger(":memory:")
    led.record(_exp(strategy="vwap_pullback", ticker="SPY"))
    led.record(_exp(strategy="opening_range_breakout", ticker="QQQ"))
    assert len(led.recent(strategy="vwap_pullback")) == 1
    assert led.recent(ticker="QQQ")[0]["strategy"] == "opening_range_breakout"
    led.close()


def test_summary_counts_by_verdict():
    led = ExperimentLedger(":memory:")
    for v in ["PASS", "PASS", "VETO", "WEAK"]:
        led.record(_exp(verdict=v))
    assert led.summary() == {"PASS": 2, "VETO": 1, "WEAK": 1}
    led.close()


def test_params_hash_is_order_independent():
    a = params_hash("s", "T", {"x": 1, "y": 2})
    b = params_hash("s", "T", {"y": 2, "x": 1})
    assert a == b
    assert a != params_hash("s", "T", {"x": 1, "y": 3})
