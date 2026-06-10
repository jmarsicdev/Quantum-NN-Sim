// quantSim — circuit diagram (butterfly/FFT wiring) + Bloch sphere row.
// Exposes CircuitDiagram and BlochRow on window.

// ————— Circuit diagram —————
// Per layer: a column of single-qubit rotation gates, then a butterfly
// entangling stage pairing qubit i with i XOR 2^(l mod log2(n)).
function CircuitDiagram({ nQubits, nLayers, activeLayer, phase, glow }) {
  const n = nQubits, L = nLayers;
  const rowH = Math.max(15, Math.min(34, 290 / n));
  const padT = 26, padB = 34, padL = 64, padR = 26;
  const colW = Math.max(96, Math.min(170, 920 / L));
  const gateW = Math.min(34, colW * 0.26), gateH = Math.min(20, rowH * 0.72);
  const W = padL + L * colW + padR;
  const H = padT + (n - 1) * rowH + padB;
  const qy = (i) => padT + i * rowH;
  const m = Math.max(1, Math.ceil(Math.log2(n)));
  const running = phase === 'running' || phase === 'paused' || phase === 'done';

  const layers = [];
  for (let l = 0; l < L; l++) {
    const x0 = padL + l * colW;
    const xGate = x0 + colW * 0.14;
    const xA = xGate + gateW + 6;          // butterfly start
    const xB = x0 + colW - 10;             // butterfly end
    const stride = Math.pow(2, l % m);
    const status = !running ? 'queued' : l < activeLayer ? 'frozen' : l === activeLayer ? 'training' : 'queued';

    const gates = [];
    for (let i = 0; i < n; i++) {
      gates.push(
        <g key={'g' + i}>
          <rect x={xGate} y={qy(i) - gateH / 2} width={gateW} height={gateH} rx="3"
            className="cd-gate" />
          {gateH >= 14 && (
            <text x={xGate + gateW / 2} y={qy(i) + gateH * 0.22} className="cd-gate-label"
              textAnchor="middle">{l % 2 === 0 ? 'Ry' : 'Rz'}</text>
          )}
        </g>
      );
    }
    const wires = [];
    for (let i = 0; i < n; i++) {
      const p = i ^ stride;
      if (p < n && p > i) {
        // crossing butterfly pair: i→p and p→i
        wires.push(<line key={'b' + i} x1={xA} y1={qy(i)} x2={xB} y2={qy(p)} className="cd-fly" />);
        wires.push(<line key={'c' + i} x1={xA} y1={qy(p)} x2={xB} y2={qy(i)} className="cd-fly" />);
        wires.push(<circle key={'d' + i} cx={xA} cy={qy(i)} r="2.6" className="cd-node" />);
        wires.push(<circle key={'e' + i} cx={xA} cy={qy(p)} r="2.6" className="cd-node" />);
        wires.push(<circle key={'f' + i} cx={xB} cy={qy(i)} r="2.6" className="cd-node" />);
        wires.push(<circle key={'h' + i} cx={xB} cy={qy(p)} r="2.6" className="cd-node" />);
      }
    }
    const statusLabel = status === 'training' ? 'training' : status === 'frozen' ? 'frozen' : 'queued';
    layers.push(
      <g key={'L' + l} className={'cd-layer cd-' + status + (glow && status === 'training' ? ' cd-glow' : '')}>
        {gates}{wires}
        <text x={x0 + colW / 2} y={H - 10} textAnchor="middle" className="cd-layer-label">
          {'L' + (l + 1) + ' · ' + statusLabel}
        </text>
        {status === 'training' && (
          <rect x={x0 + 2} y={padT - 14} width={colW - 12} height={H - padT - padB + 26}
            rx="7" className="cd-active-frame" />
        )}
      </g>
    );
  }

  const wireLabels = [];
  const labelEvery = n > 16 ? 4 : 1;
  for (let i = 0; i < n; i++) {
    if (i % labelEvery === 0 || i === n - 1) {
      wireLabels.push(
        <text key={'q' + i} x={padL - 12} y={qy(i) + 3.5} textAnchor="end" className="cd-qlabel">
          {'q' + i}
        </text>
      );
    }
  }

  return (
    <svg className="cd-svg" viewBox={'0 0 ' + W + ' ' + H} role="img"
      aria-label={'Quantum circuit, ' + n + ' qubits, ' + L + ' layers'}>
      {/* qubit rails */}
      {Array.from({ length: n }, (_, i) => (
        <line key={'r' + i} x1={padL - 4} y1={qy(i)} x2={W - padR + 4} y2={qy(i)} className="cd-rail" />
      ))}
      {wireLabels}
      {layers}
      {/* measurement ticks */}
      {Array.from({ length: n }, (_, i) => (
        <path key={'m' + i} d={'M ' + (W - padR + 4) + ' ' + (qy(i) - 4) + ' l 6 4 l -6 4'} className="cd-meas" />
      ))}
    </svg>
  );
}

// ————— Bloch sphere row —————
// Canvas-rendered pseudo-3D spheres. Arrow direction = qubit state,
// arrow LENGTH = purity (short arrow ⇒ entangled with the rest).
const BLOCH_TILT = 0.42;

function drawBloch(ctx, S, st, colors, t) {
  const cx = S / 2, cy = S / 2, R = S * 0.40;
  ctx.clearRect(0, 0, S, S);

  // sphere body
  const grad = ctx.createRadialGradient(cx - R * 0.35, cy - R * 0.4, R * 0.1, cx, cy, R * 1.05);
  grad.addColorStop(0, 'rgba(190,185,220,0.13)');
  grad.addColorStop(0.7, 'rgba(140,135,175,0.05)');
  grad.addColorStop(1, 'rgba(20,18,30,0)');
  ctx.fillStyle = grad;
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = 'rgba(205,200,235,0.22)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();

  // equator + meridian
  ctx.strokeStyle = 'rgba(205,200,235,0.13)';
  ctx.beginPath(); ctx.ellipse(cx, cy, R, R * Math.sin(BLOCH_TILT), 0, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.ellipse(cx, cy, R * Math.sin(BLOCH_TILT), R, 0, 0, Math.PI * 2); ctx.stroke();

  // z axis + poles
  ctx.strokeStyle = 'rgba(205,200,235,0.18)';
  ctx.setLineDash([2, 3]);
  ctx.beginPath(); ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R); ctx.stroke();
  ctx.setLineDash([]);

  // state vector → screen (orthographic, tilted about x)
  const sx = Math.sin(st.theta) * Math.cos(st.phi + t * 0.0001);
  const sy = Math.sin(st.theta) * Math.sin(st.phi + t * 0.0001);
  const sz = Math.cos(st.theta);
  const px = cx + R * st.r * (sx * Math.cos(0.5) + sy * Math.sin(0.5));
  const py = cy - R * st.r * (sz * Math.cos(BLOCH_TILT) - sy * Math.sin(BLOCH_TILT) * 0.6);

  // entanglement → color shift (pure: cool, entangled: warm)
  const ent = 1 - (st.r - 0.18) / 0.82;
  const col = ent > 0.55 ? colors.warn : colors.arrow;

  // arrow shaft
  ctx.strokeStyle = col;
  ctx.lineWidth = Math.max(1.4, S * 0.022);
  ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(px, py); ctx.stroke();
  // arrow head
  const ang = Math.atan2(py - cy, px - cx);
  const hl = Math.max(4, S * 0.07);
  ctx.fillStyle = col;
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.lineTo(px - hl * Math.cos(ang - 0.42), py - hl * Math.sin(ang - 0.42));
  ctx.lineTo(px - hl * Math.cos(ang + 0.42), py - hl * Math.sin(ang + 0.42));
  ctx.closePath(); ctx.fill();
  // origin dot
  ctx.fillStyle = 'rgba(205,200,235,0.4)';
  ctx.beginPath(); ctx.arc(cx, cy, 1.6, 0, Math.PI * 2); ctx.fill();
}

function BlochRow({ nQubits, targetsRef, colors, animLevel, phase }) {
  const wrapRef = React.useRef(null);
  const curRef = React.useRef([]);
  const live = phase === 'running' || phase === 'paused' || phase === 'done';

  // size spheres to fit one row
  const size = nQubits <= 8 ? 96 : nQubits <= 12 ? 78 : nQubits <= 18 ? 62 : 50;

  React.useEffect(() => {
    let raf, dead = false;
    const lerpRate = 0.04 + 0.14 * (animLevel / 100);
    function loop(t) {
      if (dead) return;
      const targets = targetsRef.current || [];
      const cur = curRef.current;
      const wrap = wrapRef.current;
      if (wrap) {
        const canvases = wrap.querySelectorAll('canvas');
        for (let i = 0; i < canvases.length; i++) {
          const tgt = targets[i] || { theta: 0.16, phi: 0, r: 1 };
          if (!cur[i]) cur[i] = { theta: tgt.theta, phi: tgt.phi, r: tgt.r };
          const c = cur[i];
          // shortest-path phi lerp
          let dphi = ((tgt.phi - c.phi + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
          c.phi += dphi * lerpRate;
          c.theta += (tgt.theta - c.theta) * lerpRate;
          c.r += (tgt.r - c.r) * lerpRate;
          // idle: slow precession so the instrument feels alive
          if (animLevel > 8) c.phi += 0.0022 * (animLevel / 60);
          const ctx = canvases[i].getContext('2d');
          drawBloch(ctx, canvases[i].width, c, colors, t);
        }
      }
      raf = requestAnimationFrame(loop);
    }
    raf = requestAnimationFrame(loop);
    return () => { dead = true; cancelAnimationFrame(raf); };
  }, [nQubits, colors, animLevel, targetsRef]);

  const tiles = [];
  for (let i = 0; i < nQubits; i++) {
    const tgt = (targetsRef.current || [])[i];
    const ent = tgt ? Math.max(0, Math.min(1, 1 - (tgt.r - 0.18) / 0.82)) : 0;
    tiles.push(
      <div className={'bloch-tile' + (live ? '' : ' is-idle')} key={i}>
        <canvas width={size} height={size} style={{ width: size + 'px', height: size + 'px' }}></canvas>
        <div className="bloch-meta">
          <span className="bloch-q">{'q' + i}</span>
          {size >= 62 && (
            <span className="bloch-ent-bar" title="entanglement">
              <span className="bloch-ent-fill" style={{
                width: (live ? Math.round(ent * 100) : 0) + '%',
                background: ent > 0.55 ? colors.warn : colors.arrow
              }}></span>
            </span>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="bloch-row" ref={wrapRef}>
      {tiles}
      {!live && <div className="bloch-idle-note">awaiting state stream</div>}
    </div>
  );
}

Object.assign(window, { CircuitDiagram, BlochRow });
