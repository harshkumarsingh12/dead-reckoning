/**
 * The auto-generated post-run report panel.
 *
 * OWNER: Akshit (presentation) + Sikruti (the numbers underneath, dr_core.eval)
 * MILESTONE: M4  |  Spec: docs/BUILD_PLAN.md sections 6.8, 8
 *
 * The moment the walk ends: an error-vs-time strip chart, an error CDF, the
 * loop-closure error in metres, and the drift percentage, with every baseline plotted
 * alongside. Worth more than any slide precisely because it visibly was not prepared
 * in advance.
 *
 * This component is a pure presenter: it renders whatever `RunReport` it is given and
 * fetches the four plot images from the gateway's `/reports/{runId}/...` route. It
 * does not decide WHEN a run has ended or HOW the report was generated -- that
 * trigger does not exist yet, and honestly can't until the ESKF is wired into the
 * live gateway and a surveyed ground-truth loop exists (see
 * `services/gateway/reports.py`'s module docstring for the full reasoning). Until
 * then, feed it a report by hand -- run `scripts/run_eval.py` on a recording, point
 * the gateway at the resulting directory with `--reports`, and pass the run's
 * `report.json` (parsed) plus its `runId` in here.
 */
export interface RunReport {
  run_id: string
  distance_m: number
  duration_s: number
  ate_m: number
  rte_60s_m: number
  final_error_m: number
  drift_pct: number
  baseline_drift_pct: Record<string, number>
  nis_consistent: Record<string, boolean>
  coverage_1sigma: number | null
  inference_ms_median: number | null
  notes: string
}

export interface PostRunPanelProps {
  report: RunReport | null
  onDismiss?: () => void
}

const PLOTS = [
  { file: 'trajectory.png', label: 'Trajectory' },
  { file: 'error_time.png', label: 'Error vs time' },
  { file: 'error_cdf.png', label: 'Error CDF' },
  { file: 'nis.png', label: 'NIS consistency' },
] as const

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="report-stat">
      <span className="strip-label">{label}</span>
      <span className="report-stat-value">{value}</span>
    </div>
  )
}

export function PostRunPanel({ report, onDismiss }: PostRunPanelProps) {
  if (!report) return null

  const reportsBase = `/reports/${report.run_id}`
  const allNisConsistent = Object.values(report.nis_consistent).every(Boolean)

  return (
    <div className="post-run-panel" data-testid="post-run-panel">
      <div className="post-run-header">
        <h2>Run report — {report.run_id}</h2>
        {onDismiss ? (
          <button type="button" className="post-run-dismiss" onClick={onDismiss}>
            Dismiss
          </button>
        ) : null}
      </div>

      <div className="report-stats">
        <Stat label="Distance" value={`${report.distance_m.toFixed(1)} m`} />
        <Stat label="Duration" value={`${report.duration_s.toFixed(0)} s`} />
        <Stat label="ATE" value={`${report.ate_m.toFixed(2)} m`} />
        <Stat label="RTE (60s)" value={`${report.rte_60s_m.toFixed(2)} m`} />
        <Stat label="Loop-closure error" value={`${report.final_error_m.toFixed(2)} m`} />
        <Stat label="Drift" value={`${report.drift_pct.toFixed(1)}%`} />
        {report.coverage_1sigma !== null ? (
          <Stat label="Coverage @1σ" value={`${(report.coverage_1sigma * 100).toFixed(0)}%`} />
        ) : null}
        {report.inference_ms_median !== null ? (
          <Stat label="Inference (median)" value={`${report.inference_ms_median.toFixed(1)} ms`} />
        ) : null}
      </div>

      {Object.keys(report.baseline_drift_pct).length > 0 ? (
        <div className="report-baselines">
          <span className="strip-label">Baselines (drift %)</span>
          <div className="report-baseline-list">
            {Object.entries(report.baseline_drift_pct).map(([name, drift]) => (
              <span className="badge badge--muted" key={name}>
                <span className="badge-dot" aria-hidden="true" />
                {name}: {drift.toFixed(1)}%
              </span>
            ))}
          </div>
        </div>
      ) : null}

      <div className="report-nis-summary">
        <span className={`badge badge--${allNisConsistent ? 'ok' : 'warn'}`}>
          <span className="badge-dot" aria-hidden="true" />
          NIS {allNisConsistent ? 'consistent on every channel' : 'inconsistent on at least one channel'}
        </span>
      </div>

      <div className="report-plots">
        {PLOTS.map(({ file, label }) => (
          <figure className="report-plot" key={file}>
            <img src={`${reportsBase}/${file}`} alt={label} loading="lazy" />
            <figcaption>{label}</figcaption>
          </figure>
        ))}
      </div>

      {report.notes ? <p className="report-notes">{report.notes}</p> : null}
    </div>
  )
}
