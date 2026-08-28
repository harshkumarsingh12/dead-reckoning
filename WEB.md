# WEB.md — the live demo UI

**Path:** `apps/web/`  |  **Owners:** Tanmay (map, socket, layout), Akshit (telemetry
strip, design system)  |  **Milestone:** M4  |  **Spec:** docs/BUILD_PLAN.md §6.8

## What this is

The screen judges actually watch. A React + Leaflet page that shows the tracked
position moving on an offline map while GPS is toggled off, with a live telemetry strip
underneath proving the filter's uncertainty is honest rather than decorative. It is the
downstream end of the chain in `docs/ARCHITECTURE.md` — it renders whatever
`services/gateway` broadcasts and touches nothing upstream of that.

**Two things this UI must never do**, both enforced structurally, not just by
convention:

1. **Never fetch a map tile from the internet.** The demo runs with the venue network
   physically unplugged. `ci-web.yml` greps the production bundle for
   `tile|basemaps|mapbox|maptiler` hostnames and **fails the build** if one appears.
2. **Never extrapolate the dot forward to hide the gateway's ~300 ms lag.** Draw the
   lagged filter state as it arrives. Extrapolating makes the dot cut corners, which
   reads as broken on stage and is dishonest about what was actually measured — see
   `docs/CONVENTIONS.md` §3 ("The lagged timeline").

## Requirements

| | |
|---|---|
| Node | 22+ (`node --version`) |
| Package manager | npm (lockfile committed) |
| Running gateway | For real data: `services/gateway` on `127.0.0.1:8000` (see `APP.md`'s sibling doc or `services/gateway/app.py`). Without it, `useTelemetry` just never receives a frame — the UI is designed to render sensibly on `frame === null`. |

## Setup

```bash
cd apps/web
npm ci
npm run dev          # http://localhost:5173, proxies /tiles /live /control to :8000
```

Other scripts: `npm run build` (typecheck + production bundle to `dist/`),
`npm run lint`, `npm run typecheck`, `npm run preview`.

`vite.config.ts` binds the dev server to every interface (`host: true`) so a phone on
the same hotspot can open the UI too, and proxies `/tiles`, `/live`, `/control` to the
gateway at `127.0.0.1:8000`. That proxy is *why* the app can use plain relative paths
everywhere — there is no hardcoded absolute host anywhere in the source to
accidentally point at the internet.

## How it works

```
App.tsx
 ├─ useTelemetry('/live')     WebSocket subscription, reconnects on drop
 │    └─ returns { frame: TelemetryFrame | null, connected: boolean }
 ├─ TrackMap        { frame }              Leaflet map: dot, ellipse, truth, baseline
 └─ TelemetryStrip  { frame, connected }   NIS gauges, ZUPT lamp, mag gate, drift %
```

### The frozen contract (`src/types.ts`)

`apps/web/src/types.ts` is the **TypeScript mirror** of `src/dr_core/types.py`. Every
field the gateway can send is typed here, and the UI is built against **hand-written
mock `TelemetryFrame` objects** rather than waiting for the ESKF (M3) to exist. That is
the entire point of the frozen contract described in `docs/ARCHITECTURE.md`: the UI
does not block on fusion, and fusion does not block on the UI.

**Changing this file is a team decision.** It must change in the same PR as
`src/dr_core/types.py`, and the group chat gets told first — CODEOWNERS requires the
repo owner's review on both sides of the contract specifically because a silent change
here breaks whoever is mid-build on the other side.

Two encoding choices worth knowing before touching this file (full reasoning in
`docs/CONVENTIONS.md` §6):

- **`t_ns` is a decimal `string`, not a `number`.** Nanoseconds since boot exceed
  `Number.MAX_SAFE_INTEGER` after roughly 104 days of device uptime, and `JSON.parse`
  silently rounds past that point. A string round-trips exactly; converting back to a
  number happens (if at all) only for display arithmetic, never for equality or ordering.
- **Enums cross the wire as their lowercase string value** (`"rejected_dip"`, not an
  integer) — they need to be debuggable from the browser console at 2 a.m., not just
  from the Python source.

### `useTelemetry` (owner: Tanmay)

Subscribes to `GET /live` (a WebSocket despite the verb — see `services/gateway`).
Reconnects on drop with a fixed backoff: the phone hotspot **will** hiccup at least
once during a real demo run, and a UI that needs a manual page refresh at that exact
moment loses the room. Implemented and verified against a real gateway — a malformed
frame is dropped rather than tearing the socket down.

### `TrackMap` (owner: Tanmay)

Implemented, real Leaflet:

1. Base tile layer from `GET /tiles/{z}/{x}/{y}.png` — the gateway's local MBTiles
   route, never an external host (`ci-web.yml` fails the build if one shows up in the
   shipped bundle).
2. The raw-integration **baseline** dot (`frame.baseline_p_world`) — the one that
   visibly spirals away. Watching this diverge in real time next to the tracked dot is,
   per the build plan, "the most persuasive thing on screen."
3. The estimated dot (`frame.state.p_world`) with a 1-sigma uncertainty ellipse, drawn
   from an eigendecomposition of `frame.state.cov`'s position sub-block
   (`map/enu.ts`).
4. The known true path (`frame.truth_p_world`), when available.

Every marker is a Leaflet `circleMarker` (plain SVG), not `L.marker` with the default
icon — sidesteps the classic broken-icon-path problem bundlers have with Leaflet's
shipped image assets. The map is created once and never recentres itself after the
first frame, so it never fights an operator's manual pan/zoom mid-demo.

Converting `p_world` (local ENU metres) back to real lat/lng for the basemap needed a
small, additive, backward-compatible extension to the frozen contract:
`TelemetryFrame.origin_lat_deg`/`origin_lon_deg` (default `null` — nothing existing
broke). Verified against a real, freshly started gateway: pushed real GPS fixes over
`/ingest`, watched the dot's on-screen position genuinely change between two
screenshots of the same live page.

### `TelemetryStrip` (owner: Akshit)

Implemented. Always visible under the map:

- Per-channel NIS against its chi-square bounds (`frame.nis`, `frame.nis_bounds`)
- ZUPT / ZARU firing lamp (`frame.zupt_active`, `frame.zaru_active`)
- Magnetometer gate verdict, with the *specific* rejection reason (`frame.mag_verdict`)
- The model's current 1-sigma (`frame.model_sigma_mps`)
- The current heading source (`frame.state.heading_source`)
- The live drift-% counter (`frame.drift_pct`) — the headline number

This is the answer to "how do you know your uncertainty is honest?" before a judge
finishes asking it — see `docs/DEMO_RUNBOOK.md`'s Q&A section.

### `PostRunPanel` (owner: Akshit, presentation + Sikruti, the numbers)

The auto-generated post-run report — stats (distance, duration, ATE, RTE, loop-closure
error, drift %, coverage@1σ, inference time), per-baseline drift badges, a NIS-
consistency summary, and the four plots `generate_report` writes, fetched from the
gateway's `GET /reports/{run_id}/{file}`. Implemented as a pure presenter: it renders
whatever `RunReport` it is handed and does not decide when a run has ended or how the
report was generated. Not currently mounted in `App.tsx` — there is no live signal for
"a run just ended" yet, since that needs the ESKF wired into the live path plus a
surveyed ground-truth loop (see `services/gateway/reports.py`). Feed it a report by
hand today: run `scripts/run_eval.py` on a recording, point the gateway at the
resulting directory with `--reports`, and pass the parsed `report.json` in as a prop.

### `ui/tokens.ts` (owner: Akshit)

One source of truth for colour, spacing, radius. Two constraints that are not taste:

1. **Projected in a bright hall.** Low-contrast greys that look refined on a laptop
   screen become invisible on a projector.
2. **Accept / reject / warn must not rely on hue alone.** Roughly one judge in twelve
   is colour-blind — pair colour with shape or a label.

## Current status

| Piece | Status |
|---|---|
| `types.ts` (frozen contract mirror) | ✅ complete, matches `dr_core.types` — now including `origin_lat_deg`/`origin_lon_deg` |
| App shell, component wiring | ✅ implemented |
| Vite config, dev proxy, offline-tile-host CI guard | ✅ implemented |
| `useTelemetry` — real WebSocket logic | ✅ implemented, verified against a live gateway |
| `TrackMap` — Leaflet, dot, ellipse, baseline | ✅ implemented, verified against a live gateway |
| `TelemetryStrip` — NIS/ZUPT/mag-gate/drift UI | ✅ implemented, verified against a mock frame |
| `PostRunPanel` — the auto-generated report | ✅ implemented, verified against a real gateway + real report files |
| `npm run lint` / `typecheck` / `build` | ✅ green, both locally and in `ci-web.yml` |

The web app is feature-complete for M4. The gateway side it connects to (`WS /live`,
`GET /tiles`, `GET /reports`, `POST /control/gps`) is fully implemented too — see
`services/gateway/app.py` and `services/gateway/hub.py`. What's left is real-world, not
code: `/live`'s broadcast is still a flat-earth GPS passthrough placeholder (no ESKF
wired into the live path yet), and there is no live trigger for the report panel
(needs that same ESKF plus a surveyed ground-truth loop) — both documented in
`services/gateway/hub.py` and `services/gateway/reports.py`. Neither blocks this UI
from being built and tested today; both are M3-and-beyond follow-ups.

## Related

- `docs/CONVENTIONS.md` §6 — wire format, timestamp/enum encoding
- `docs/ARCHITECTURE.md` — the frozen-contract philosophy this app is built on
- `docs/DEMO_RUNBOOK.md` — what the strip needs to show and why, in judging terms
- `services/gateway/hub.py` — what `/live` actually sends right now, and its limits
- Issues #38 (map), #39 (telemetry strip), #40 (post-run report panel)
