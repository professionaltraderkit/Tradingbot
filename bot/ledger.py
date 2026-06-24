"""The experiment ledger: a durable record of every validation run.

This is the compounding half of the loop. Each time the Checker (validation.py)
rules on a parameter set, the params, the data window, the out-of-sample metrics,
the multiple-testing bar and the verdict land here. That makes the search
*stateful*: you can pull the best PASS for a strategy to deploy, see what's
already been tried and why it failed (so you don't re-burn compute on known dead
ends), and watch an edge decay across monthly re-runs - instead of rediscovering
the same conclusions every time.

SQLite, same write-through style as bot/state.py. Default file: experiments.db
(alongside tradingbot.db, and likewise gitignored); pass ':memory:' in tests.
"""

import hashlib
import json
import sqlite3
import subprocess
from datetime import datetime, timezone

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    strategy TEXT NOT NULL,
    ticker TEXT NOT NULL,
    name TEXT,
    timeframe TEXT,
    config_file TEXT,
    data_start TEXT,
    data_end TEXT,
    bars INTEGER,
    cost_bps REAL,
    n_trials INTEGER,
    n_folds INTEGER,
    params TEXT,
    params_hash TEXT NOT NULL,
    is_return REAL,
    oos_return REAL,
    oos_trades INTEGER,
    oos_tstat REAL,
    threshold_t REAL,
    gap REAL,
    consistency REAL,
    verdict TEXT NOT NULL,
    reasons TEXT,
    git_sha TEXT
);
CREATE INDEX IF NOT EXISTS idx_exp_lookup ON experiments (strategy, ticker, verdict);
CREATE INDEX IF NOT EXISTS idx_exp_hash ON experiments (params_hash);
"""

# Columns written by record(), in order. created_at and params_hash are derived.
_COLUMNS = (
    "created_at", "strategy", "ticker", "name", "timeframe", "config_file",
    "data_start", "data_end", "bars", "cost_bps", "n_trials", "n_folds",
    "params", "params_hash", "is_return", "oos_return", "oos_trades", "oos_tstat",
    "threshold_t", "gap", "consistency", "verdict", "reasons", "git_sha",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def params_hash(strategy: str, ticker: str, params: dict) -> str:
    """Stable id for a (strategy, ticker, params) triple, so identical configs
    collapse to one searchable key regardless of dict ordering."""
    blob = json.dumps({"s": strategy, "t": ticker, "p": params or {}},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def current_git_sha() -> str | None:
    """Short HEAD sha so each row is pinned to the code that produced it. Best
    effort - None outside a repo or without git."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


class ExperimentLedger:
    def __init__(self, db_path: str = "experiments.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def record(self, exp: dict) -> int:
        """Write one experiment; returns its row id. `exp` carries the metrics
        (see validation.ledger_row); created_at and params_hash are filled here."""
        row = {
            "created_at": _now(),
            "strategy": exp["strategy"],
            "ticker": exp["ticker"],
            "name": exp.get("name"),
            "timeframe": exp.get("timeframe"),
            "config_file": exp.get("config_file"),
            "data_start": exp.get("data_start"),
            "data_end": exp.get("data_end"),
            "bars": exp.get("bars"),
            "cost_bps": exp.get("cost_bps"),
            "n_trials": exp.get("n_trials"),
            "n_folds": exp.get("n_folds"),
            "params": json.dumps(exp.get("params") or {}, sort_keys=True, default=str),
            "params_hash": exp.get("params_hash") or params_hash(
                exp["strategy"], exp["ticker"], exp.get("params") or {}),
            "is_return": exp.get("is_return"),
            "oos_return": exp.get("oos_return"),
            "oos_trades": exp.get("oos_trades"),
            "oos_tstat": exp.get("oos_tstat"),
            "threshold_t": exp.get("threshold_t"),
            "gap": exp.get("gap"),
            "consistency": exp.get("consistency"),
            "verdict": exp["verdict"],
            "reasons": json.dumps(exp.get("reasons") or [], default=str),
            "git_sha": exp.get("git_sha"),
        }
        placeholders = ", ".join(f":{c}" for c in _COLUMNS)
        cur = self.conn.execute(
            f"INSERT INTO experiments ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
            row,
        )
        self.conn.commit()
        return cur.lastrowid

    def recent(self, limit: int = 20, strategy: str | None = None,
               ticker: str | None = None) -> list[dict]:
        """Most recent experiments first, optionally filtered."""
        clauses, args = [], []
        if strategy:
            clauses.append("strategy = ?"); args.append(strategy)
        if ticker:
            clauses.append("ticker = ?"); args.append(ticker)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        args.append(limit)
        rows = self.conn.execute(
            f"SELECT * FROM experiments{where} ORDER BY id DESC LIMIT ?", args
        ).fetchall()
        return [self._decode(r) for r in rows]

    def best(self, strategy: str, ticker: str, verdict: str = "PASS") -> dict | None:
        """The deploy candidate: highest out-of-sample return among rows with the
        given verdict (PASS by default) for this strategy+ticker."""
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE strategy=? AND ticker=? AND verdict=? "
            "ORDER BY oos_return DESC LIMIT 1", (strategy, ticker, verdict)
        ).fetchone()
        return self._decode(row) if row else None

    def seen(self, strategy: str, ticker: str, params: dict) -> dict | None:
        """The most recent ruling on this exact config, if it's been tried."""
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE params_hash=? ORDER BY id DESC LIMIT 1",
            (params_hash(strategy, ticker, params),)
        ).fetchone()
        return self._decode(row) if row else None

    def summary(self) -> dict[str, int]:
        """Verdict -> count across the whole ledger."""
        rows = self.conn.execute(
            "SELECT verdict, COUNT(*) AS n FROM experiments GROUP BY verdict"
        ).fetchall()
        return {r["verdict"]: r["n"] for r in rows}

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["params"] = json.loads(d["params"]) if d.get("params") else {}
        d["reasons"] = json.loads(d["reasons"]) if d.get("reasons") else []
        return d

    def close(self) -> None:
        self.conn.close()
