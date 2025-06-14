from live import broker as br
from live.broker import Broker, BrokerConfig

class DummyClient:
    def futures_change_leverage(self, *a, **kw):
        pass
    def futures_mark_price(self, symbol):
        return {"markPrice": "30000"}
    def futures_create_order(self, **kw):
        return kw


def test_open_close_dryrun(monkeypatch):
    monkeypatch.setattr(br, "Client", lambda k, s: DummyClient())
    monkeypatch.setattr(br, "load_keys", lambda: ("k", "s"))
    cfg = BrokerConfig("BTCUSDT", 1, dry_run=True, starting_equity=100)
    b = Broker(cfg)
    b.last_kline = {"close": 30000}
    open_resp = b.open_long()
    assert open_resp["side"] == "BUY"
    close_resp = b.close_position()
    assert close_resp["side"] == "SELL"
