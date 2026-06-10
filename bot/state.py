"""SQLite persistence: open positions, closed trades, equity, event log.

Everything is written through immediately so the bot can be killed and
restarted without losing track of open paper positions.
"""

import sqlite3
from datetime import datetime, timezone

_SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    ticker TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    entry_price REAL NOT NULL,
    stop_price REAL NOT NULL,
    risk_amount REAL NOT NULL,
    opened_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    name TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    pnl REAL NOT NULL,
    reason TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS account (
    key TEXT PRIMARY KEY,
    value REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    kind TEXT NOT NULL,
    message TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateStore:
    def __init__(self, db_path: str, starting_equity: float):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.execute(
            "INSERT OR IGNORE INTO account (key, value) VALUES ('cash', ?)",
            (starting_equity,),
        )
        self.conn.commit()

    # -- account ----------------------------------------------------------
    def get_cash(self) -> float:
        row = self.conn.execute("SELECT value FROM account WHERE key='cash'").fetchone()
        return row["value"]

    def set_cash(self, value: float) -> None:
        self.conn.execute("UPDATE account SET value=? WHERE key='cash'", (value,))
        self.conn.commit()

    # -- positions ---------------------------------------------------------
    def open_positions(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM positions").fetchall()
        return [dict(r) for r in rows]

    def get_position(self, ticker: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM positions WHERE ticker=?", (ticker,)
        ).fetchone()
        return dict(row) if row else None

    def save_position(self, pos: dict) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO positions
               (ticker, name, side, quantity, entry_price, stop_price, risk_amount, opened_at)
               VALUES (:ticker, :name, :side, :quantity, :entry_price, :stop_price,
                       :risk_amount, :opened_at)""",
            pos,
        )
        self.conn.commit()

    def delete_position(self, ticker: str) -> None:
        self.conn.execute("DELETE FROM positions WHERE ticker=?", (ticker,))
        self.conn.commit()

    # -- trades ------------------------------------------------------------
    def record_trade(self, trade: dict) -> None:
        self.conn.execute(
            """INSERT INTO trades
               (ticker, name, side, quantity, entry_price, exit_price, pnl,
                reason, opened_at, closed_at)
               VALUES (:ticker, :name, :side, :quantity, :entry_price, :exit_price,
                       :pnl, :reason, :opened_at, :closed_at)""",
            trade,
        )
        self.conn.commit()

    def trades_since(self, iso_timestamp: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM trades WHERE closed_at >= ? ORDER BY closed_at",
            (iso_timestamp,),
        ).fetchall()
        return [dict(r) for r in rows]

    def all_trades(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM trades ORDER BY closed_at").fetchall()
        return [dict(r) for r in rows]

    # -- events --------------------------------------------------------------
    def log_event(self, kind: str, message: str) -> None:
        self.conn.execute(
            "INSERT INTO events (at, kind, message) VALUES (?, ?, ?)",
            (_now(), kind, message),
        )
        self.conn.commit()

    def events_since(self, iso_timestamp: str, kind: str | None = None) -> list[dict]:
        if kind:
            rows = self.conn.execute(
                "SELECT * FROM events WHERE at >= ? AND kind = ? ORDER BY at",
                (iso_timestamp, kind),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM events WHERE at >= ? ORDER BY at", (iso_timestamp,)
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()
