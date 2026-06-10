// quantSim — simulation engine + WebSocket-shaped transport.
// The UI talks to SimSocket exactly as it would a real WebSocket
// (readyState / onopen / onmessage / send(JSON)). Swap SimSocket for
// `new WebSocket(url)` and the dashboard works against a live backend.
(function () {
  'use strict';

  var SHOT_STOPS = [32, 128, 512, 2048, 8192, Infinity];

  // Deterministic PRNG so runs are reproducible per-config.
  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

  function QuantSimEngine(cfg) { this.reset(cfg); }

  QuantSimEngine.prototype.reset = function (cfg) {
    var prev = this.cfg || {};
    this.cfg = {
      nQubits: cfg && cfg.nQubits != null ? cfg.nQubits : (prev.nQubits || 8),
      nLayers: cfg && cfg.nLayers != null ? cfg.nLayers : (prev.nLayers || 3),
      shots: cfg && cfg.shots !== undefined ? cfg.shots : (prev.shots !== undefined ? prev.shots : 2048),
      epochsPerLayer: 36
    };
    var n = this.cfg.nQubits, L = this.cfg.nLayers;
    this.epoch = 0;
    this.done = false;
    this.totalEpochs = L * this.cfg.epochsPerLayer;
    this.layerStarts = [];
    for (var l = 0; l < L; l++) this.layerStarts.push(l * this.cfg.epochsPerLayer);
    this.loss = { exact: [], naive: [], par: [] };
    this.grads = [];                       // per epoch: array[L] of magnitude | null
    this.budget = { exact: 0, naive: 0, par: 0 };
    this._rand = mulberry32(1337 + n * 101 + L * 7);
    this._lossFrom = 0.92 + 0.06 * this._rand();
    this._lastExact = this._lossFrom;
    this._layerStartLoss = this._lossFrom;
    this._walk = { naive: 0, par: 0 };
    // Bloch state per qubit: theta/phi (direction), r (purity: 1 = pure, short = entangled)
    this.bloch = [];
    for (var i = 0; i < n; i++) {
      this.bloch.push({
        theta: 0.18 + 0.25 * this._rand(),
        phi: this._rand() * Math.PI * 2,
        r: 1,
        _w1: 0.6 + this._rand() * 0.9,      // drift personality
        _w2: this._rand() * Math.PI * 2,
        _ent: 0.45 + 0.55 * this._rand()    // how entangled this qubit ends up
      });
    }
  };

  QuantSimEngine.prototype.activeLayer = function () {
    return Math.min(this.cfg.nLayers - 1, Math.floor(this.epoch / this.cfg.epochsPerLayer));
  };

  // Barren-plateau factor: gradients vanish exponentially in qubit count.
  QuantSimEngine.prototype.bpFactor = function () {
    return Math.exp(-0.115 * (this.cfg.nQubits - 2));
  };

  QuantSimEngine.prototype._lossFloor = function (layersTrained) {
    return 0.035 + 0.88 * Math.pow(0.40, layersTrained);
  };

  QuantSimEngine.prototype.tick = function () {
    if (this.done) return this.frame();
    var cfg = this.cfg, rand = this._rand;
    var n = cfg.nQubits, L = cfg.nLayers, E = cfg.epochsPerLayer;
    var l = this.activeLayer();
    var e = this.epoch - this.layerStarts[l];
    if (e === 0) this._layerStartLoss = this._lastExact;

    var bp = this.bpFactor();
    // Effective learning speed collapses on a plateau (that's the lesson).
    var rate = 0.155 * (0.06 + 0.94 * bp);
    var target = this._lossFloor(l + 1);
    var prog = 1 - Math.exp(-rate * (e + 1));
    var lossExact = target + (this._layerStartLoss - target) * (1 - prog);
    lossExact = clamp(lossExact, 0.02, 1.05);
    this._lastExact = lossExact;

    // Finite shots ⇒ noisy loss estimates for the parameter-shift modes.
    // Naive evaluates each parameter independently (noisier); the parallelized
    // mode batches shifts and averages over commuting groups (tracks exact closely).
    var sigmaN = cfg.shots === Infinity ? 0 : 0.62 / Math.sqrt(cfg.shots);
    var sigmaP = sigmaN * 0.38;
    this._walk.naive = this._walk.naive * 0.84 + (rand() - 0.5) * sigmaN;
    this._walk.par = this._walk.par * 0.78 + (rand() - 0.5) * sigmaP;
    var lossNaive = clamp(lossExact + this._walk.naive + (rand() - 0.5) * sigmaN * 0.7, 0.012, 1.1);
    var lossPar = clamp(lossExact + this._walk.par + (rand() - 0.5) * sigmaP * 0.7, 0.012, 1.1);
    this.loss.exact.push(lossExact);
    this.loss.naive.push(lossNaive);
    this.loss.par.push(lossPar);

    // Gradient magnitudes per layer (null = not yet trained / not measured).
    var g = [];
    var depthFade = Math.exp(-0.30 * l);
    var shotFloor = cfg.shots === Infinity ? 0 : 0.004 / Math.sqrt(cfg.shots / 512);
    for (var j = 0; j < L; j++) {
      if (j < l) {
        g.push(Math.max(1e-6, 1.6e-4 * bp * (0.5 + rand())));            // frozen: residual
      } else if (j === l) {
        var live = (0.055 + 0.62 * Math.exp(-rate * 5.4 * (e + 1) / E)) * bp * depthFade;
        live *= (1 + 0.38 * (rand() - 0.5));
        g.push(Math.max(1e-6, live + shotFloor * rand()));
      } else {
        g.push(null);
      }
    }
    this.grads.push(g);

    // Hardware budget (circuit executions per optimizer step):
    //   exact autodiff: 1 analytic pass (simulator only)
    //   naive parameter-shift: 2 evaluations per parameter
    //   parallelized parameter-shift: shifts batched across commuting groups
    var P = 3 * n;                                  // params in the active layer
    var groups = Math.max(1, Math.ceil(P / n));     // ≈3 commuting groups
    this.budget.exact += 1;
    this.budget.naive += 2 * P;
    this.budget.par += 2 * groups;

    // Bloch evolution: arrows precess; purity drops as entanglement builds.
    var trainedFrac = (this.epoch + 1) / this.totalEpochs;
    for (var q = 0; q < n; q++) {
      var b = this.bloch[q];
      b.phi = (b.phi + 0.22 * b._w1 + 0.1 * (rand() - 0.5)) % (Math.PI * 2);
      var thTarget = 0.5 + 0.95 * Math.sin(trainedFrac * 2.4 * b._w1 + b._w2) * 0.5 + 0.45;
      b.theta = clamp(b.theta + (thTarget - b.theta) * 0.10 + (rand() - 0.5) * 0.07, 0.12, Math.PI - 0.12);
      var entangle = clamp(trainedFrac * b._ent * 1.15, 0, 1);
      var rTarget = 1 - 0.68 * entangle;
      b.r = clamp(b.r + (rTarget - b.r) * 0.16 + (rand() - 0.5) * 0.015, 0.18, 1);
    }

    this.epoch++;
    if (this.epoch >= this.totalEpochs) this.done = true;
    return this.frame();
  };

  QuantSimEngine.prototype.frame = function () {
    var latest = this.loss.exact.length - 1;
    return {
      type: 'tick',
      epoch: this.epoch,
      totalEpochs: this.totalEpochs,
      activeLayer: this.activeLayer(),
      done: this.done,
      loss: latest >= 0 ? {
        exact: this.loss.exact[latest],
        naive: this.loss.naive[latest],
        par: this.loss.par[latest]
      } : null,
      grads: latest >= 0 ? this.grads[latest] : null,
      bloch: this.bloch.map(function (b) { return { theta: b.theta, phi: b.phi, r: b.r }; }),
      budget: { exact: this.budget.exact, naive: this.budget.naive, par: this.budget.par }
    };
  };

  // ——— WebSocket-shaped transport over the local engine ———
  function SimSocket(url) {
    var self = this;
    this.url = url;
    this.readyState = 0;                  // CONNECTING
    this.onopen = null; this.onmessage = null; this.onclose = null;
    this.engine = new QuantSimEngine({});
    this._running = false;
    this._speed = 3;                      // epochs / sec
    this._timer = null;
    this._connectTimer = setTimeout(function () { self._open(); }, 1100);
  }

  SimSocket.prototype._open = function () {
    this.readyState = 1;                  // OPEN
    if (this.onopen) this.onopen();
    this._emit({ type: 'hello', server: 'quantsim-sim/0.4.2', protocol: 1 });
    this._emit(this._configMsg());
  };

  SimSocket.prototype._configMsg = function () {
    return {
      type: 'config',
      nQubits: this.engine.cfg.nQubits,
      nLayers: this.engine.cfg.nLayers,
      shots: this.engine.cfg.shots === Infinity ? 'inf' : this.engine.cfg.shots,
      epochsPerLayer: this.engine.cfg.epochsPerLayer,
      totalEpochs: this.engine.totalEpochs,
      layerStarts: this.engine.layerStarts.slice(),
      bloch: this.engine.frame().bloch
    };
  };

  SimSocket.prototype._emit = function (obj) {
    if (this.readyState !== 1 || !this.onmessage) return;
    this.onmessage({ data: JSON.stringify(obj) });
  };

  SimSocket.prototype._startTimer = function () {
    var self = this;
    this._stopTimer();
    this._timer = setInterval(function () {
      if (!self._running || self.readyState !== 1) return;
      self._emit(self.engine.tick());
      if (self.engine.done) {
        self._running = false;
        self._emit({ type: 'run_complete' });
        self._stopTimer();
      }
    }, 1000 / this._speed);
  };
  SimSocket.prototype._stopTimer = function () {
    if (this._timer) { clearInterval(this._timer); this._timer = null; }
  };

  SimSocket.prototype.send = function (raw) {
    if (this.readyState !== 1) return;
    var msg = typeof raw === 'string' ? JSON.parse(raw) : raw;
    switch (msg.cmd) {
      case 'configure': {
        var shots = msg.shots === 'inf' ? Infinity : msg.shots;
        this.engine.reset({ nQubits: msg.nQubits, nLayers: msg.nLayers, shots: shots });
        this._running = false;
        this._stopTimer();
        this._emit(this._configMsg());
        break;
      }
      case 'start':
        if (this.engine.done) { this.engine.reset({}); this._emit(this._configMsg()); }
        this._running = true;
        this._startTimer();
        this._emit({ type: 'run_started' });
        break;
      case 'pause':
        this._running = false;
        this._emit({ type: 'run_paused' });
        break;
      case 'step':
        this._running = false;
        this._emit(this.engine.tick());
        if (this.engine.done) this._emit({ type: 'run_complete' });
        break;
      case 'reset':
        this._running = false;
        this._stopTimer();
        this.engine.reset({});
        this._emit(this._configMsg());
        this._emit({ type: 'run_reset' });
        break;
      case 'speed':
        this._speed = clamp(msg.value || 3, 0.25, 12);
        if (this._running) this._startTimer();
        break;
    }
  };

  // Educational extra: simulate a dropped connection (Tweaks panel button).
  SimSocket.prototype.simulateDrop = function () {
    var self = this;
    if (this.readyState !== 1) return;
    var wasRunning = this._running;
    this.readyState = 3;                  // CLOSED
    this._running = false;
    this._stopTimer();
    if (this.onclose) this.onclose({ code: 1006, reason: 'simulated network drop' });
    setTimeout(function () {
      self.readyState = 1;
      if (self.onopen) self.onopen();
      self._emit({ type: 'hello', server: 'quantsim-sim/0.4.2', protocol: 1, resumed: true });
      self._emit(self._configMsg());
      // replay state so the UI can resume seamlessly
      self._emit({ type: 'resume', frame: self.engine.frame(), history: self.engine.loss, grads: self.engine.grads });
      if (wasRunning && !self.engine.done) {
        self._running = true;
        self._startTimer();
        self._emit({ type: 'run_started' });
      }
    }, 3200);
  };

  SimSocket.prototype.close = function () {
    this.readyState = 3;
    this._stopTimer();
    clearTimeout(this._connectTimer);
  };

  // Live transport: a real WebSocket to the quantsim backend, with the same
  // object-accepting send() the UI uses against SimSocket.
  function LiveSocket(url) {
    var ws = new WebSocket(url);
    var nativeSend = ws.send.bind(ws);
    var queue = [];
    ws.addEventListener('open', function () {
      while (queue.length) nativeSend(queue.shift());
    });
    // Mirror SimSocket's tolerance: the UI fires commands (e.g. initial speed)
    // before the connection opens; queue them instead of throwing.
    ws.send = function (m) {
      var s = typeof m === 'string' ? m : JSON.stringify(m);
      if (ws.readyState === 0) queue.push(s);
      else if (ws.readyState === 1) nativeSend(s);
    };
    ws.simulateDrop = function () { ws.close(); };
    return ws;
  }

  window.QuantSim = { QuantSimEngine: QuantSimEngine, SimSocket: SimSocket, LiveSocket: LiveSocket, SHOT_STOPS: SHOT_STOPS };
})();
