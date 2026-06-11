// quantSim — model analysis feed, epoch timeline scrubber, end-of-run report.
// Exposes AnalysisFeed, TimelineScrubber, ReportOverlay on window.

const PANEL_TAGS = { circuit: 'circuit', bloch: 'qubits', loss: 'loss', heat: 'plateau', budget: 'budget' };

// ————— Analysis feed (sidebar) —————
// Walks tick frames newest-first up to the displayed epoch, so scrubbing the
// timeline replays the commentary exactly as it appeared.
function AnalysisFeed({ frames, uptoEpoch, live }) {
  if (!live) return <div className="feed-idle mono">commentary starts with the first epoch</div>;
  const items = [];
  for (let e = Math.min(uptoEpoch, frames.length); e >= 1 && items.length < 8; e--) {
    const a = frames[e - 1] && frames[e - 1].analysis;
    if (!a || !a.observations) continue;
    for (const o of a.observations) {
      items.push({ epoch: e, panel: o.panel, text: o.text, tone: o.tone });
      if (items.length >= 8) break;
    }
  }
  if (!items.length) return <div className="feed-idle mono">no observations yet — keep training</div>;
  return (
    <div className="feed">
      {items.map((it, i) => (
        <div className={'feed-item tone-' + it.tone} key={it.epoch + '-' + it.panel + '-' + i}>
          <div className="feed-meta mono">
            <span className="feed-epoch">e{String(it.epoch).padStart(3, '0')}</span>
            <span className="feed-chip">{PANEL_TAGS[it.panel] || it.panel}</span>
            <span className="feed-tone">{it.tone}</span>
          </div>
          <div className="feed-text">{it.text}</div>
        </div>
      ))}
    </div>
  );
}

// ————— Timeline scrubber (run control) —————
function TimelineScrubber({ frames, viewEpoch, onScrub, onLive, disabled }) {
  const n = frames.length;
  const viewing = viewEpoch != null;
  const value = viewing ? Math.min(viewEpoch, n) : n;
  const off = disabled || !n;
  return (
    <div className={'scrub' + (off ? ' is-disabled' : '')}>
      <div className="ctl-label-row">
        <span className="ctl-label">timeline</span>
        <span className="ctl-value mono">{!n ? '—' : viewing ? 'epoch ' + value + ' · replay' : 'live'}</span>
      </div>
      <input type="range" className="ctl-slider" min={1} max={Math.max(1, n)} step={1}
        value={Math.max(1, value)} disabled={off}
        onChange={(e) => onScrub(Number(e.target.value))} />
      <div className="scrub-btns">
        <button className="btn btn-mini" disabled={off || value <= 1}
          onClick={() => onScrub(value - 1)}>‹ prev</button>
        <button className="btn btn-mini" disabled={off || !viewing}
          onClick={() => (value >= n ? onLive() : onScrub(value + 1))}>next ›</button>
        <button className="btn btn-mini" disabled={off || !viewing} onClick={onLive}>● live</button>
      </div>
      <div className="ctl-caption">drag to replay any epoch — every panel re-renders that moment</div>
    </div>
  );
}

// ————— End-of-run report overlay —————
function ReportOverlay({ report, onClose, onReplay }) {
  if (!report) return null;
  const cfg = report.config;
  const modeColor = { exact: 'var(--c-exact)', naive: 'var(--c-naive)', par: 'var(--c-par)' };
  const fmtInt = (v) => (v == null ? '—' : Number(v).toLocaleString('en-US'));
  const fmtPct = (v) => (v == null ? '—' : Math.round(v * 100) + '%');
  return (
    <div className="report-veil" onClick={onClose} data-screen-label="Training report">
      <div className="report-card" onClick={(e) => e.stopPropagation()}>
        <header className="report-top">
          <div>
            <div className="report-kicker mono">training report</div>
            <h2 className="report-title">{report.headline}</h2>
          </div>
          <button className="btn btn-mini" onClick={onClose} aria-label="Close report">✕</button>
        </header>
        <div className="report-cfg mono">
          {cfg.nQubits} qubits · {cfg.nLayers} layers · {cfg.totalParams} trainable circuit params ·
          shots {cfg.shots === 'inf' ? '∞' : cfg.shots} · {cfg.totalEpochs} epochs ·
          mean qubit purity {report.bloch.meanPurity.toFixed(2)}
          {report.bloch.entropy != null ? ' · cut entropy ' + report.bloch.entropy.toFixed(2) + ' bits' : ''}
        </div>

        <h3 className="report-h">Same model, three trainers</h3>
        <table className="report-table">
          <thead>
            <tr><th>method</th><th>final loss</th><th>train accuracy</th><th>circuit executions</th></tr>
          </thead>
          <tbody>
            {QS_MODES.map((m) => {
              const row = report.modes[m.key];
              return (
                <tr key={m.key}>
                  <td>
                    <span className="legend-swatch" style={{ background: modeColor[m.key], marginRight: 8 }}></span>
                    {m.label}
                  </td>
                  <td className="mono">{qsFmt(row.finalLoss)}</td>
                  <td className="mono">{fmtPct(row.accuracy)}</td>
                  <td className="mono">{fmtInt(row.executions)}
                    {m.key === 'naive' && report.budget.naiveOverPar > 1
                      ? ' · ×' + report.budget.naiveOverPar.toFixed(1) + ' vs parallel' : ''}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>

        <h3 className="report-h">Layer by layer</h3>
        <table className="report-table">
          <thead>
            <tr><th>layer</th><th>loss start → end</th><th>improvement</th><th>peak ‖∇θ‖</th><th>verdict</th></tr>
          </thead>
          <tbody>
            {report.layers.map((c) => (
              <tr key={c.layer}>
                <td className="mono">L{c.layer + 1}</td>
                <td className="mono">{qsFmt(c.startLoss)} → {qsFmt(c.endLoss)}</td>
                <td className="mono">{c.deltaPct >= 0 ? '−' : '+'}{Math.abs(c.deltaPct).toFixed(1)}%</td>
                <td className="mono">{qsFmt(c.peakGrad)}</td>
                <td>{c.plateau
                  ? <span className="heat-warn">barren plateau</span>
                  : <span className="report-ok mono">healthy gradients</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <h3 className="report-h">Takeaways</h3>
        <ul className="report-takes">
          {report.takeaways.map((t, i) => <li key={i}>{t}</li>)}
        </ul>

        <footer className="report-foot">
          <button className="btn btn-primary" onClick={onReplay}>⟲ step through the run</button>
          <button className="btn" onClick={onClose}>close</button>
        </footer>
      </div>
    </div>
  );
}

Object.assign(window, { AnalysisFeed, TimelineScrubber, ReportOverlay });
