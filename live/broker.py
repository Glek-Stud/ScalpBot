"""Thin wrapper around Binance REST API."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from binance.client import Client  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Client = None
try:
    from binance.exceptions import BinanceAPIException  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BinanceAPIException = Exception  # type: ignore


def load_keys(path: str | None = None):
    """Return Binance API keys from an ``ipynb`` file inside the repo."""
    import json
    from pathlib import Path

    if path is None:
        root = Path(__file__).resolve().parents[1]
        path = root / "collect" / "Binance_keys.ipynb"
    p = Path(path)
    with p.open() as f:
        nb = json.load(f)

    src = "".join(nb["cells"][0]["source"])
    key = secret = None
    for line in src.splitlines():
        if "api_key" in line:
            key = line.split("=", 1)[1].strip().strip("'\"").replace("\\n", "")
        if "api_secret" in line:
            secret = line.split("=", 1)[1].strip().strip("'\"")
    return key, secret


@dataclass
class BrokerConfig:
    symbol: str
    leverage: int
    dry_run: bool = True
    starting_equity: float = 1000.0
    usd_per_trade: float = 10.0


class Broker:
    def __init__(self, cfg: BrokerConfig):
        if Client is None:
            raise ImportError("python-binance package is required to use Broker")
        key, sec = load_keys()
        self.client = Client(key, sec)
        self.cfg = cfg
        self.position = 0
        self.last_kline = None
        self.qty = 0.0
        try:
            if not cfg.dry_run:
                self.client.futures_change_leverage(
                    symbol=cfg.symbol,
                    leverage=cfg.leverage,
                    recvWindow=5000,
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # helper methods
    # ------------------------------------------------------------------

    def _mark_price(self) -> float:
        """Return current mark price for configured symbol."""
        try:
            data = self.client.futures_mark_price(symbol=self.cfg.symbol)
            return float(data["markPrice"])
        except Exception:
            return float(self.last_kline["close"])

    def _calc_qty(self, price: float) -> float:
        usd = self.cfg.usd_per_trade * self.cfg.leverage
        if not self.cfg.dry_run:
            try:
                balances = self.client.futures_account_balance()
                for b in balances:
                    if b.get("asset") == "USDT":
                        avail = float(b.get("availableBalance", 0.0)) * self.cfg.leverage
                        usd = min(usd, avail)
                        break
            except Exception:
                pass
        qty = usd / price
        # Round to six decimals and clamp to Binance minimum lot size
        qty = max(qty, 0.001)
        return float(f"{qty:.6f}")

    # ------------------------------------------------------------------

    def update_kline(self, kline: dict):
        self.last_kline = {
            "close": float(kline["k"]["c"]),
            "open": float(kline["k"]["o"]),
            "high": float(kline["k"]["h"]),
            "low": float(kline["k"]["l"]),
            "volume": float(kline["k"]["v"]),
        }

    def execute(self, desired_pos: int):
        """Flatten then open position according to ``desired_pos``."""
        if desired_pos == self.position:
            return 0.0, self.position

        price = self._mark_price()
        qty = self._calc_qty(price)

        if self.cfg.dry_run:
            if self.position != 0:
                side = "SELL" if self.position > 0 else "BUY"
                print(f"[DRY-RUN] close {qty} {self.cfg.symbol} @ {price}")
            if desired_pos != 0:
                side = "BUY" if desired_pos > 0 else "SELL"
                print(f"[DRY-RUN] open {qty} {self.cfg.symbol} @ {price}")
            self.position = desired_pos
            self.qty = qty if desired_pos != 0 else 0.0
            return 0.0, self.position

        if self.position != 0:
            side = "SELL" if self.position > 0 else "BUY"
            try:
                self.client.futures_create_order(
                    symbol=self.cfg.symbol,
                    side=side,
                    type="MARKET",
                    quantity=self.qty,
                    recvWindow=5000,
                    reduceOnly=True,
                )
            except Exception:
                pass

        if desired_pos != 0:
            side = "BUY" if desired_pos > 0 else "SELL"
            try:
                self.client.futures_create_order(
                    symbol=self.cfg.symbol,
                    side=side,
                    type="MARKET",
                    quantity=qty,
                    recvWindow=5000,
                )
            except Exception:
                pass
            self.qty = qty
        else:
            self.qty = 0.0

        self.position = desired_pos
        return 0.0, self.position

    # ------------------------------------------------------------------
    # manual trade helpers
    # ------------------------------------------------------------------

    def open_long(self) -> dict:
        price = self._mark_price()
        qty = self._calc_qty(price)
        self.qty = qty
        if self.cfg.dry_run:
            print(f"[DRY-RUN] BUY {qty} {self.cfg.symbol} @ {price}")
            self.position = 1
            return {"side": "BUY", "qty": qty, "price": price}
        try:
            resp = self.client.futures_create_order(
                symbol=self.cfg.symbol,
                side="BUY",
                type="MARKET",
                quantity=qty,
                recvWindow=5000,
            )
            self.position = 1
            return resp
        except BinanceAPIException as e:
            return {"error": str(e)}

    def open_short(self) -> dict:
        price = self._mark_price()
        qty = self._calc_qty(price)
        self.qty = qty
        if self.cfg.dry_run:
            print(f"[DRY-RUN] SELL {qty} {self.cfg.symbol} @ {price}")
            self.position = -1
            return {"side": "SELL", "qty": qty, "price": price}
        try:
            resp = self.client.futures_create_order(
                symbol=self.cfg.symbol,
                side="SELL",
                type="MARKET",
                quantity=qty,
                recvWindow=5000,
            )
            self.position = -1
            return resp
        except BinanceAPIException as e:
            return {"error": str(e)}

    def close_position(self) -> dict | None:
        if self.position == 0:
            print("No position to close")
            return None
        price = self._mark_price()
        qty = self.qty
        side = "SELL" if self.position > 0 else "BUY"
        if self.cfg.dry_run:
            print(f"[DRY-RUN] {side} {qty} {self.cfg.symbol} @ {price}")
            self.position = 0
            self.qty = 0.0
            return {"side": side, "qty": qty, "price": price}
        try:
            resp = self.client.futures_create_order(
                symbol=self.cfg.symbol,
                side=side,
                type="MARKET",
                quantity=qty,
                reduceOnly=True,
                recvWindow=5000,
            )
            self.position = 0
            self.qty = 0.0
            return resp
        except BinanceAPIException as e:
            return {"error": str(e)}
