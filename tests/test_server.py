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

        # first epoch carries commentary from the analysis engine
        analysis = tick["analysis"]
        assert isinstance(analysis["headline"], str)
        assert analysis["observations"]
        for o in analysis["observations"]:
            assert o["panel"] in {"circuit", "bloch", "loss", "heat", "budget"}
            assert o["tone"] in {"info", "insight", "warning", "milestone"}


def test_run_to_completion_sends_report():
    client = TestClient(app)
    with client.websocket_connect("/quantsim/v1") as ws:
        ws.receive_json()  # hello
        ws.receive_json()  # default config
        ws.send_json({"cmd": "configure", "nQubits": 4, "nLayers": 1, "shots": "inf"})
        config = ws.receive_json()
        total = config["totalEpochs"]

        for i in range(total):
            ws.send_json({"cmd": "step"})
            tick = ws.receive_json()
            assert tick["type"] == "tick" and tick["epoch"] == i + 1

        report_msg = ws.receive_json()
        assert report_msg["type"] == "report"
        rep = report_msg["report"]
        assert isinstance(rep["headline"], str)
        assert set(rep["modes"]) == {"exact", "naive", "par"}
        for mode in rep["modes"].values():
            assert 0.0 <= mode["accuracy"] <= 1.0
        assert len(rep["layers"]) == 1
        assert rep["budget"]["naiveOverPar"] > 1
        assert rep["bloch"]["entropy"] is not None
        assert len(rep["takeaways"]) >= 4

        assert ws.receive_json()["type"] == "run_complete"


def test_index_serves_dashboard():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "quantSim" in r.text
