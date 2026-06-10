// quantSim — app shell: socket wiring, state machine, layout, sidebar, tweaks.
const { useState, useEffect, useRef, useCallback } = React;

const SHOT_STOPS = window.QuantSim.SHOT_STOPS;
const SHOT_LABELS = ['32', '128', '512', '2048', '8192', '∞'];

const PALETTES = [
  ['#38bdf8', '#fb7185', '#fbbf24'],   // ember (exact / naive / parallel+active)
  ['#22d3ee', '#a78bfa', '#f0abfc'],   // ion
  ['#34d399', '#a3e635', '#fbbf24']    // phosphor
];

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "palette": ["#38bdf8", "#fb7185", "#fbbf24"],
  "anim": 60,
  "density": "balanced",
  "glow": true,
  "speed": 3
}/*EDITMODE-END*/;

const INFO = {
  circuit: {
    title: 'The circuit being trained',
    body: 'Each column is one trainable layer: a rotation gate (Ry/Rz) on every qubit, then an entangling stage wired in a butterfly (FFT-like) pattern — qubit i pairs with qubit i⊕2ˡ, so information mixes across the whole register in log₂(n) layers. Training is layer-wise: the glowing layer is learning right now, dimmed layers are frozen at their trained values, faint layers are queued.'
  },
  bloch: {
    title: 'Bloch spheres — one per qubit',
    body: 'A single qubit\'s state is a point on (or inside) a sphere: |0⟩ at the north pole, |1⟩ at the south. The arrow shows where each qubit currently points. Arrow LENGTH is purity: as a qubit entangles with the others, its individual state becomes mixed and the arrow shrinks toward the center. Short warm-colored arrows = strongly entangled.'
  },
  loss: {
    title: 'Three ways to compute the same gradient',
    body: 'All three modes minimize the same loss. Exact autodiff backpropagates through a classical simulation — smooth, but impossible on real hardware. Parameter-shift estimates each gradient by running the circuit with parameters nudged ±π/2, so it works on a real device but inherits shot noise. Vertical markers show where training hands off to the next layer.'
  },
  heat: {
    title: 'Barren plateau monitor',
    body: 'Each cell is the gradient magnitude ‖∇θ‖ of one layer at one epoch. In deep or wide random circuits, gradients vanish exponentially with qubit count — the dreaded barren plateau. Drag the qubit slider up and watch the map fade toward violet: when the active layer drops below 10⁻³, training has effectively stalled.'
  },
  budget: {
    title: 'What a gradient step costs on hardware',
    body: 'Real quantum computers bill by circuit execution. Exact autodiff needs one analytic pass (simulator only). Naive parameter-shift runs the circuit twice per parameter — 2 × 3n executions per step. Parallelizing shifts across commuting parameter groups collapses that to a handful. Same gradients, wildly different bills.'
  },
  controls: {
    title: 'Experiment controls',
    body: 'Qubits and layers reshape the circuit (and reset the run). The shot slider sets how many times each circuit is measured — fewer shots means noisier gradients; ∞ is a perfect, noiseless simulator. Step advances exactly one epoch so you can read every panel between updates.'
  }
};

function InfoDot({ id, openId, setOpenId }) {
  const open = openId === id;
  return (
    <span className="info-wrap">
      <button className={'info-dot' + (open ? ' is-open' : '')} aria-label={'About: ' + INFO[id].title}
        onClick={(e) => { e.stopPropagation(); setOpenId(open ? null : id); }}>i</button>
      {open && (
        <div className="info-pop" onClick={(e) => e.stopPropagation()}>
          <div className="info-pop-title">{INFO[id].title}</div>
          <div className="info-pop-body">{INFO[id].body}</div>
        </div>
      )}
    </span>
  );
}

function Panel({ id, title, meta, children, openId, setOpenId, className, screenLabel }) {
  return (
    <section className={'panel ' + (className || '')} data-screen-label={screenLabel || title}>
      <header className="panel-head">
        <h2 className="panel-title">{title}</h2>
        {meta && <div className="panel-meta">{meta}</div>}
        <InfoDot id={id} openId={openId} setOpenId={setOpenId} />
      </header>
      <div className="panel-body">{children}</div>
    </section>
  );
}

function Skeleton({ h }) {
  return <div className="skel" style={{ height: h }}></div>;
}

function SliderRow({ label, caption, value, display, min, max, step, onChange, disabled }) {
  return (
    <div className={'ctl-row' + (disabled ? ' is-disabled' : '')}>
      <div className="ctl-label-row">
        <span className="ctl-label">{label}</span>
        <span className="ctl-value mono">{display}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))} className="ctl-slider" />
      {caption && <div className="ctl-caption">{caption}</div>}
    </div>
  );
}

function App() {
  const [t, setTweak] = window.useTweaks(TWEAK_DEFAULTS);
  const pal = Array.isArray(t.palette) ? t.palette : PALETTES[0];
  const colors = {
    exact: pal[0], naive: pal[1], par: pal[2],
    arrow: pal[0], warn: pal[1], heat: pal[2],
    heatRGB: (() => {
      const h = pal[2].replace('#', '');
      return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
    })()
  };

  // ——— connection + run state ———
  const [phase, setPhase] = useState('connecting'); // connecting | idle | running | paused | done | disconnected
  const phaseRef = useRef(phase);
  phaseRef.current = phase;
  const [cfg, setCfg] = useState({ nQubits: 8, nLayers: 3, shotIdx: 3 });
  const [run, setRun] = useState({
    epoch: 0, totalEpochs: 108, activeLayer: 0,
    layerStarts: [0, 36, 72],
    histories: { exact: [], naive: [], par: [] },
    grads: [],
    budget: { exact: 0, naive: 0, par: 0 }
  });
  const blochRef = useRef([]);
  const sockRef = useRef(null);
  const [openInfo, setOpenInfo] = useState(null);

  useEffect(() => {
    // Live backend when served over http(s) (opt out with ?live=0); mock engine otherwise.
    const live = location.protocol.startsWith('http') &&
      new URLSearchParams(location.search).get('live') !== '0';
    const sock = live
      ? new window.QuantSim.LiveSocket(
          (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/quantsim/v1')
      : new window.QuantSim.SimSocket('wss://localhost:8765/quantsim/v1');
    sockRef.current = sock;
    sock.onopen = () => {};
    sock.onclose = () => setPhase('disconnected');
    sock.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      switch (msg.type) {
        case 'hello':
          if (!msg.resumed) setPhase('idle');
          break;
        case 'config':
          blochRef.current = msg.bloch;
          setPhase('idle');
          setRun({
            epoch: 0, totalEpochs: msg.totalEpochs, activeLayer: 0,
            layerStarts: msg.layerStarts,
            histories: { exact: [], naive: [], par: [] },
            grads: [], budget: { exact: 0, naive: 0, par: 0 }
          });
          break;
        case 'run_started': setPhase('running'); break;
        case 'run_paused': setPhase('paused'); break;
        case 'run_complete': setPhase('done'); break;
        case 'run_reset': setPhase('idle'); break;
        case 'resume': {
          const f = msg.frame;
          blochRef.current = f.bloch;
          setRun((r) => ({
            ...r, epoch: f.epoch, totalEpochs: f.totalEpochs, activeLayer: f.activeLayer,
            histories: { exact: msg.history.exact.slice(), naive: msg.history.naive.slice(), par: msg.history.par.slice() },
            grads: msg.grads.slice(), budget: f.budget
          }));
          setPhase(f.done ? 'done' : f.epoch > 0 ? 'paused' : 'idle');
          break;
        }
        case 'tick': {
          blochRef.current = msg.bloch;
          setRun((r) => {
            const h = r.histories;
            if (msg.loss) {
              h.exact = h.exact.concat(msg.loss.exact);
              h.naive = h.naive.concat(msg.loss.naive);
              h.par = h.par.concat(msg.loss.par);
            }
            return {
              ...r, epoch: msg.epoch, totalEpochs: msg.totalEpochs, activeLayer: msg.activeLayer,
              histories: { exact: h.exact, naive: h.naive, par: h.par },
              grads: msg.grads ? r.grads.concat([msg.grads]) : r.grads,
              budget: msg.budget
            };
          });
          if (phaseRef.current !== 'running' && phaseRef.current !== 'done') setPhase('paused');
          if (msg.done) setPhase('done');
          break;
        }
      }
    };
    const closeInfo = () => setOpenInfo(null);
    document.addEventListener('click', closeInfo);
    return () => { sock.close(); document.removeEventListener('click', closeInfo); };
  }, []);

  // tweak side-effects
  useEffect(() => {
    const r = document.documentElement;
    r.style.setProperty('--c-exact', colors.exact);
    r.style.setProperty('--c-naive', colors.naive);
    r.style.setProperty('--c-par', colors.par);
    r.style.setProperty('--anim-scale', String(t.anim / 100));
    r.classList.toggle('density-dense', t.density === 'dense');
    r.classList.toggle('no-pulse', t.anim <= 10);
  }, [pal.join(','), t.anim, t.density]);

  useEffect(() => {
    if (sockRef.current) sockRef.current.send({ cmd: 'speed', value: t.speed });
  }, [t.speed]);

  const send = (cmd, extra) => {
    if (sockRef.current && sockRef.current.readyState === 1) {
      sockRef.current.send(Object.assign({ cmd }, extra));
    }
  };

  const reconfigure = (next) => {
    setCfg(next);
    send('configure', {
      nQubits: next.nQubits, nLayers: next.nLayers,
      shots: SHOT_STOPS[next.shotIdx] === Infinity ? 'inf' : SHOT_STOPS[next.shotIdx]
    });
  };

  const connecting = phase === 'connecting';
  const disconnected = phase === 'disconnected';
  const live = phase === 'running' || phase === 'paused' || phase === 'done';
  const shotsLabel = SHOT_LABELS[cfg.shotIdx] === '∞' ? '∞ · ideal' : SHOT_LABELS[cfg.shotIdx] + ' shots';

  const statusChip = connecting
    ? { cls: 'chip-wait', txt: 'connecting…' }
    : disconnected
      ? { cls: 'chip-bad', txt: 'reconnecting…' }
      : { cls: 'chip-ok', txt: 'connected' };

  return (
    <div className="shell">
      {/* ——— header ——— */}
      <header className="topbar" data-screen-label="Header">
        <div className="brand">
          <span className="brand-mark mono">quant<span className="brand-accent">Sim</span></span>
          <span className="brand-tag">quantum neural network · training visualizer</span>
        </div>
        <div className="run-readout mono">
          {live ? (
            <React.Fragment>
              <span>epoch <b>{String(run.epoch).padStart(3, '0')}</b>/{run.totalEpochs}</span>
              <span className="sep">·</span>
              <span>layer <b>L{run.activeLayer + 1}</b>/{cfg.nLayers}</span>
              <span className="sep">·</span>
              <span className={'run-state run-' + phase}>{phase === 'done' ? 'complete' : phase}</span>
            </React.Fragment>
          ) : (
            <span className="run-state-dim">{connecting ? 'initializing' : disconnected ? 'link down' : 'standing by'}</span>
          )}
        </div>
        <div className={'ws-chip ' + statusChip.cls}>
          <span className="ws-dot"></span>
          <span className="mono">wss://localhost:8765 · {statusChip.txt}</span>
        </div>
      </header>

      {disconnected && (
        <div className="banner-drop" data-screen-label="Disconnected banner">
          <span className="ws-dot"></span>
          connection lost — retrying… last data shown frozen below
        </div>
      )}

      <div className="layout">
        <main className="main-col">
          {/* ——— circuit hero ——— */}
          <Panel id="circuit" title="Circuit topology" screenLabel="Circuit panel"
            className="panel-hero"
            meta={<span className="mono">{cfg.nQubits} qubits · {cfg.nLayers} layers · butterfly wiring</span>}
            openId={openInfo} setOpenId={setOpenInfo}>
            {connecting ? <Skeleton h={300} /> : (
              <React.Fragment>
                <CircuitDiagram nQubits={cfg.nQubits} nLayers={cfg.nLayers}
                  activeLayer={run.activeLayer} phase={phase} glow={t.glow} />
                {phase === 'idle' && <div className="hero-hint mono">▸ press play to begin layer-wise training</div>}
                {phase === 'done' && <div className="hero-hint hero-done mono">run complete — all layers trained</div>}
              </React.Fragment>
            )}
          </Panel>

          {/* ——— bloch row ——— */}
          <Panel id="bloch" title="Qubit states" screenLabel="Bloch panel"
            meta={<span className="mono">arrow length = purity · short = entangled</span>}
            openId={openInfo} setOpenId={setOpenInfo}>
            {connecting ? <Skeleton h={130} /> : (
              <BlochRow nQubits={cfg.nQubits} targetsRef={blochRef} colors={colors}
                animLevel={disconnected ? 0 : t.anim} phase={phase} />
            )}
          </Panel>

          {/* ——— loss + heatmap ——— */}
          <div className="duo">
            <Panel id="loss" title="Training loss" screenLabel="Loss panel"
              meta={<span className="mono">{shotsLabel}</span>}
              openId={openInfo} setOpenId={setOpenInfo}>
              {connecting ? <Skeleton h={280} /> : (
                <LossChart histories={run.histories} layerStarts={run.layerStarts}
                  totalEpochs={run.totalEpochs} epoch={run.epoch} colors={colors} phase={phase} />
              )}
            </Panel>
            <Panel id="heat" title="Barren plateau monitor" screenLabel="Plateau panel"
              meta={<span className="mono">‖∇θ‖ / layer / epoch</span>}
              openId={openInfo} setOpenId={setOpenInfo}>
              {connecting ? <Skeleton h={200} /> : (
                <GradHeatmap grads={run.grads} nLayers={cfg.nLayers} totalEpochs={run.totalEpochs}
                  layerStarts={run.layerStarts} colors={colors} phase={phase} nQubits={cfg.nQubits} />
              )}
            </Panel>
          </div>

          {/* ——— budget odometer ——— */}
          <Panel id="budget" title="Hardware budget" screenLabel="Budget panel"
            meta={<span className="mono">circuit executions this run</span>}
            openId={openInfo} setOpenId={setOpenInfo}>
            {connecting ? <Skeleton h={120} /> : (
              <BudgetOdometer budget={run.budget} colors={colors} phase={phase} nQubits={cfg.nQubits} />
            )}
          </Panel>
        </main>

        {/* ——— sidebar ——— */}
        <aside className="sidebar" data-screen-label="Controls sidebar">
          <section className="panel">
            <header className="panel-head">
              <h2 className="panel-title">Run control</h2>
              <InfoDot id="controls" openId={openInfo} setOpenId={setOpenInfo} />
            </header>
            <div className="panel-body">
              <div className="transport">
                <button className="btn btn-primary" disabled={connecting || disconnected}
                  onClick={() => send(phase === 'running' ? 'pause' : 'start')}>
                  {phase === 'running' ? '❚❚ pause' : phase === 'done' ? '▸ rerun' : '▸ play'}
                </button>
                <button className="btn" disabled={connecting || disconnected || phase === 'running' || phase === 'done'}
                  onClick={() => send('step')}>step</button>
                <button className="btn" disabled={connecting || disconnected || !live}
                  onClick={() => send('reset')}>reset</button>
              </div>

              <div className="ctl-divider"></div>

              <SliderRow label="qubits" value={cfg.nQubits} min={2} max={28} step={1}
                display={cfg.nQubits} disabled={connecting || disconnected}
                caption="wider registers → flatter gradients (barren plateau)"
                onChange={(v) => reconfigure({ ...cfg, nQubits: v })} />
              <SliderRow label="layers" value={cfg.nLayers} min={1} max={6} step={1}
                display={cfg.nLayers} disabled={connecting || disconnected}
                caption="trained one at a time, left to right"
                onChange={(v) => reconfigure({ ...cfg, nLayers: v })} />
              <SliderRow label="hardware noise" value={cfg.shotIdx} min={0} max={5} step={1}
                display={shotsLabel} disabled={connecting || disconnected}
                caption="shots per measurement · ∞ = noiseless simulator"
                onChange={(v) => reconfigure({ ...cfg, shotIdx: v })} />

              <div className="ctl-note mono">changing topology resets the run</div>
            </div>
          </section>
        </aside>
      </div>

      {/* ——— tweaks ——— */}
      <window.TweaksPanel>
        <window.TweakSection label="Instrument" />
        <window.TweakColor label="Palette" value={t.palette} options={PALETTES}
          onChange={(v) => setTweak('palette', v)} />
        <window.TweakSlider label="Animation" value={t.anim} min={0} max={100} step={10}
          onChange={(v) => setTweak('anim', v)} />
        <window.TweakToggle label="Layer glow" value={t.glow} onChange={(v) => setTweak('glow', v)} />
        <window.TweakRadio label="Density" value={t.density} options={['balanced', 'dense']}
          onChange={(v) => setTweak('density', v)} />
        <window.TweakSection label="Simulation" />
        <window.TweakSlider label="Speed" value={t.speed} min={1} max={10} step={1} unit=" eps"
          onChange={(v) => setTweak('speed', v)} />
        <window.TweakButton label="Simulate disconnect"
          onClick={() => sockRef.current && sockRef.current.simulateDrop()} />
      </window.TweaksPanel>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
