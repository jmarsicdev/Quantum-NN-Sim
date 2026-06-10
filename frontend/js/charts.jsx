// quantSim — loss chart, barren-plateau heatmap, hardware-budget odometer.
// Exposes LossChart, GradHeatmap, BudgetOdometer on window.

const QS_MODES = [
  { key: 'exact', label: 'exact autodiff', sub: 'simulator only' },
  { key: 'naive', label: 'naive parameter-shift', sub: '2 runs / parameter' },
  { key: 'par', label: 'parallelized parameter-shift', sub: 'batched shifts' }
];

function qsFmt(v) {
  if (v == null || isNaN(v)) return '—';
  return v >= 0.1 ? v.toFixed(3) : v.toExponential(1);
}

// ————— Loss chart (log-y, layer-start markers) —————
function LossChart({ histories, layerStarts, totalEpochs, epoch, colors, phase }) {
  const W = 760, H = 286;
  const padL = 52, padR = 14, padT = 18, padB = 30;
  const iw = W - padL - padR, ih = H - padT - padB;
  const yMin = 0.02, yMax = 1.05;
  const ly = (v) => padT + (Math.log10(yMax) - Math.log10(Math.max(yMin, v))) /
    (Math.log10(yMax) - Math.log10(yMin)) * ih;
  const lx = (e) => padL + (e / Math.max(1, totalEpochs - 1)) * iw;
  const live = phase === 'running' || phase === 'paused' || phase === 'done';

  const yTicks = [1, 0.5, 0.2, 0.1, 0.05, 0.02];
  const modeColor = { exact: colors.exact, naive: colors.naive, par: colors.par };

  const path = (arr) => {
    if (!arr || !arr.length) return '';
    let d = '';
    for (let i = 0; i < arr.length; i++) {
      d += (i === 0 ? 'M' : 'L') + lx(i).toFixed(1) + ' ' + ly(arr[i]).toFixed(1) + ' ';
    }
    return d;
  };

  return (
    <div className="loss-wrap">
      <svg className="loss-svg" viewBox={'0 0 ' + W + ' ' + H} role="img" aria-label="Training loss curves">
        {yTicks.map((v) => (
          <g key={v}>
            <line x1={padL} y1={ly(v)} x2={W - padR} y2={ly(v)} className="ch-grid" />
            <text x={padL - 8} y={ly(v) + 3.5} textAnchor="end" className="ch-tick">{v}</text>
          </g>
        ))}
        <text x={14} y={padT + ih / 2} className="ch-axis" transform={'rotate(-90 14 ' + (padT + ih / 2) + ')'}
          textAnchor="middle">loss (log)</text>
        <text x={padL + iw / 2} y={H - 6} textAnchor="middle" className="ch-axis">epoch</text>

        {/* layer-start markers */}
        {live && layerStarts.map((s, i) => (
          <g key={'ls' + i}>
            <line x1={lx(s)} y1={padT - 4} x2={lx(s)} y2={padT + ih} className="ch-layer-mark" />
            <text x={lx(s) + 4} y={padT + 6} className="ch-layer-label">{'L' + (i + 1)}</text>
          </g>
        ))}

        {live && ['naive', 'par', 'exact'].map((k) => (
          <path key={k} d={path(histories[k])} fill="none" stroke={modeColor[k]}
            strokeWidth={k === 'exact' ? 2.2 : 1.5} opacity={k === 'exact' ? 1 : 0.85}
            strokeLinejoin="round" />
        ))}

        {/* playhead */}
        {live && epoch > 0 && epoch < totalEpochs && (
          <line x1={lx(epoch - 1)} y1={padT} x2={lx(epoch - 1)} y2={padT + ih} className="ch-playhead" />
        )}

        {!live && (
          <text x={padL + iw / 2} y={padT + ih / 2} textAnchor="middle" className="ch-empty">
            no run data — press play
          </text>
        )}
      </svg>
      <div className="loss-legend">
        {QS_MODES.map((m) => {
          const arr = histories[m.key];
          const cur = live && arr && arr.length ? arr[arr.length - 1] : null;
          return (
            <div className="legend-item" key={m.key}>
              <span className="legend-swatch" style={{ background: modeColor[m.key] }}></span>
              <span className="legend-name">{m.label}</span>
              <span className="legend-val mono">{qsFmt(cur)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ————— Barren plateau monitor (per-layer gradient heatmap) —————
function GradHeatmap({ grads, nLayers, totalEpochs, layerStarts, colors, phase, nQubits }) {
  const canvasRef = React.useRef(null);
  const live = phase === 'running' || phase === 'paused' || phase === 'done';
  const rowH = Math.max(18, Math.min(34, 140 / nLayers));
  const CW = 560, CH = nLayers * rowH;

  // log color scale: 1e-5 → 1e0  (dark = vanishing, bright = healthy)
  const colorFor = (g) => {
    if (g == null) return 'rgba(255,255,255,0.025)';
    const t = Math.max(0, Math.min(1, (Math.log10(g) + 5) / 5));
    if (t < 0.42) {                                  // vanishing zone → deep violet
      const k = t / 0.42;
      return 'rgba(' + Math.round(38 + 30 * k) + ',' + Math.round(20 + 20 * k) + ',' + Math.round(64 + 40 * k) + ',1)';
    }
    const k = (t - 0.42) / 0.58;
    // violet → accent
    const c = colors.heatRGB;
    return 'rgba(' + Math.round(68 + (c[0] - 68) * k) + ',' + Math.round(40 + (c[1] - 40) * k) + ',' +
      Math.round(104 + (c[2] - 104) * k) + ',1)';
  };

  React.useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, CW, CH);
    const cellW = CW / totalEpochs;
    for (let e = 0; e < totalEpochs; e++) {
      const col = grads[e];
      for (let l = 0; l < nLayers; l++) {
        ctx.fillStyle = colorFor(col ? col[l] : null);
        ctx.fillRect(e * cellW, l * rowH + 1, Math.ceil(cellW) + 0.5, rowH - 2);
      }
    }
  }, [grads.length, nLayers, totalEpochs, colors, phase]);

  const latest = grads.length ? grads[grads.length - 1] : null;
  const activeGrad = latest ? latest.filter((g) => g != null).slice(-1)[0] : null;
  const plateau = live && activeGrad != null && activeGrad < 1e-3;

  return (
    <div className="heat-wrap">
      <div className="heat-grid" style={{ gridTemplateRows: 'repeat(' + nLayers + ', ' + rowH + 'px)' }}>
        <div className="heat-rows">
          {Array.from({ length: nLayers }, (_, l) => (
            <div className="heat-rowlabel mono" key={l} style={{ height: rowH + 'px' }}>{'L' + (l + 1)}</div>
          ))}
        </div>
        <canvas ref={canvasRef} width={CW} height={CH} className="heat-canvas"
          style={{ height: CH + 'px' }}></canvas>
        <div className="heat-vals">
          {Array.from({ length: nLayers }, (_, l) => (
            <div className="heat-val mono" key={l} style={{ height: rowH + 'px' }}>
              {live && latest ? qsFmt(latest[l]) : '—'}
            </div>
          ))}
        </div>
      </div>
      <div className="heat-foot">
        <div className="heat-scale">
          <span className="mono">10⁻⁵</span>
          <span className="heat-scale-bar" style={{
            background: 'linear-gradient(90deg, rgb(38,20,64), rgb(68,40,104), ' + colors.heat + ')'
          }}></span>
          <span className="mono">10⁰</span>
          <span className="heat-scale-label">‖∇θ‖ per layer</span>
        </div>
        {plateau ? (
          <div className="heat-warn">⚠ plateau — gradients &lt; 10⁻³ at {nQubits} qubits</div>
        ) : (
          <div className="heat-ok">{live ? 'gradients healthy' : 'awaiting gradients'}</div>
        )}
      </div>
    </div>
  );
}

// ————— Hardware budget odometer —————
function OdoDigits({ value, places }) {
  const s = String(Math.min(value, Math.pow(10, places) - 1)).padStart(places, '0');
  const firstSig = s.search(/[1-9]/);
  return (
    <div className="odo">
      {s.split('').map((d, i) => (
        <span className={'odo-cell' + (firstSig === -1 || i < firstSig ? ' odo-dim' : '')} key={i}>
          <span className="odo-strip" style={{ transform: 'translateY(' + (-Number(d) * 10) + '%)' }}>
            {'0123456789'.split('').map((n) => <span className="odo-num" key={n}>{n}</span>)}
          </span>
        </span>
      ))}
    </div>
  );
}

function BudgetOdometer({ budget, colors, phase, nQubits }) {
  const live = phase === 'running' || phase === 'paused' || phase === 'done';
  const modeColor = { exact: colors.exact, naive: colors.naive, par: colors.par };
  const perEpoch = { exact: 1, naive: 6 * nQubits, par: 6 };
  const max = Math.max(1, budget.naive);
  return (
    <div className="budget-row">
      {QS_MODES.map((m) => {
        const v = live ? budget[m.key] : 0;
        const ratio = budget.exact > 0 ? v / budget.exact : 0;
        return (
          <div className="budget-card" key={m.key} style={{ '--mode-c': modeColor[m.key] }}>
            <div className="budget-head">
              <span className="legend-swatch" style={{ background: modeColor[m.key] }}></span>
              <span className="budget-name">{m.label}</span>
            </div>
            <OdoDigits value={v} places={8} />
            <div className="budget-sub mono">
              {live
                ? '+' + perEpoch[m.key] + ' /epoch' + (m.key !== 'exact' && ratio > 1 ? ' · ×' + Math.round(ratio) + ' vs exact' : ' · analytic pass')
                : 'idle'}
            </div>
            <div className="budget-bar">
              <div className="budget-fill" style={{
                width: (live ? Math.max(1.5, (v / max) * 100) : 0) + '%',
                background: modeColor[m.key]
              }}></div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

Object.assign(window, { LossChart, GradHeatmap, BudgetOdometer, QS_MODES, qsFmt });
