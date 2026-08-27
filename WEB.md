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
moment loses the room. Currently a stub — the hook exists and is wired into `App.tsx`,
but the actual socket-open/parse/reconnect logic is not yet implemented (`frame`
permanently `null`, `connected` permanently `false`).

### `TrackMap` (owner: Tanmay)

Not yet implemented. Will render, in order of most to least dramatic on stage:

1. Leaflet base layer from `GET /tiles/{z}/{x}/{y}.png` — the gateway's local MBTiles
   route, never an external host.
2. The raw-integration **baseline** dot (`frame.baseline_p_world`) — the one that
   visibly spirals away. Watching this diverge in real time next to the tracked dot is,
   per the build plan, "the most persuasive thing on screen."
3. The estimated dot (`frame.state.p_world`) with a 1-sigma uncertainty ellipse derived
   from `frame.state.cov`.
4. The known true path (`frame.truth_p_world`), when available.

### `TelemetryStrip` (owner: Akshit)

Not yet implemented beyond a connection-status line and the raw drift percentage. Will
carry, always visible under the map:

- Per-channel NIS against its chi-square bounds (`frame.nis`, `frame.nis_bounds`)
- ZUPT / ZARU firing lamp (`frame.zupt_active`, `frame.zaru_active`)
- Magnetometer gate verdict, with the *specific* rejection reason (`frame.mag_verdict`)
- The model's current 1-sigma (`frame.model_sigma_mps`)
- The current heading source (`frame.state.heading_source`)
- The live drift-% counter (`frame.drift_pct`) — the headline number

This is the answer to "how do you know your uncertainty is honest?" before a judge
finishes asking it — see `docs/DEMO_RUNBOOK.md`'s Q&A section.

### `ui/tokens.ts` (owner: Akshit)

One source of truth for colour, spacing, radius. Two constraints that are not taste:

1. **Projected in a bright hall.** Low-contrast greys that look refined on a laptop
   screen become invisible on a projector.
2. **Accept / reject / warn must not rely on hue alone.** Roughly one judge in twelve
   is colour-blind — pair colour with shape or a label.

## Current status

| Piece | Status |
|---|---|
| `types.ts` (frozen contract mirror) | ✅ complete, matches `dr_core.types` |
| App shell, component wiring | ✅ implemented |
| Vite config, dev proxy, offline-tile-host CI guard | ✅ implemented |
| `useTelemetry` — real WebSocket logic | ❌ stub (owner: Tanmay) |
| `TrackMap` — Leaflet, dot, ellipse, baseline | ❌ stub (owner: Tanmay) |
| `TelemetryStrip` — NIS/ZUPT/mag-gate/drift UI | ❌ stub (owner: Akshit) |
| `npm run lint` / `typecheck` / `build` | ✅ green, both locally and in `ci-web.yml` |

The gateway side this UI will eventually connect to (`WS /live`, `GET /tiles`,
`POST /control/gps`) is already implemented — see `services/gateway/app.py` and
`services/gateway/hub.py`. Its `/live` broadcast is currently a flat-earth GPS
passthrough placeholder (no ESKF yet), which is enough real, moving data for this UI to
be built and tested against today without waiting on M3.

## Related

- `docs/CONVENTIONS.md` §6 — wire format, timestamp/enum encoding
- `docs/ARCHITECTURE.md` — the frozen-contract philosophy this app is built on
- `docs/DEMO_RUNBOOK.md` — what the strip needs to show and why, in judging terms
- `services/gateway/hub.py` — what `/live` actually sends right now, and its limits
- Issues #38 (map), #39 (telemetry strip), #40 (post-run report panel)
