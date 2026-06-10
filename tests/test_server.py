from fastapi.testclient import TestClient

from quantsim.server import app


def test_websocket_protocol_handshake_and_step():
    client = TestClient(app)
    with client.websocket_connect("/quantsim/v1") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["protocol"] == 1

        config = ws.receive_json()
        assert config["type"] == "config"
        assert config["nQubits"] == 8 and len(config["bloch"]) == 8

        # reconfigure to a tiny session so the test is fast
        ws.send_json({"cmd": "configure", "nQubits": 4, "nLayers": 2, "shots": "inf"})
        config = ws.receive_json()
        assert config["nQubits"] == 4 and config["nLayers"] == 2
        assert config["shots"] == "inf"
        assert config["layerStarts"] == [0, 12]

        ws.send_json({"cmd": "step"})
        tick = ws.receive_json()
        assert tick["type"] == "tick"
        assert tick["epoch"] == 1 and tick["activeLayer"] == 0
        assert set(tick["loss"]) == {"exact", "naive", "par"}
        assert tick["grads"][0] is not None and tick["grads"][1] is None
        assert len(tick["bloch"]) == 4
        for b in tick["bloch"]:
            assert 0.0 <= b["r"] <= 1.0 + 1e-6
        # the headline: parallel-shift uses fewer executions than naive
        assert tick["budget"]["par"] < tick["budget"]["naive"]


def test_index_serves_dashboard():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "quantSim" in r.text
