/**
 * Design tokens. One source of truth for colour, spacing and type.
 *
 * OWNER: Akshit  |  MILESTONE: M4
 *
 * Two constraints that are not aesthetic preferences:
 *   1. This is projected in a bright hall. Low-contrast greys that look refined on a
 *      laptop become invisible on a projector.
 *   2. Accept / reject / warn must be distinguishable without relying on hue alone —
 *      shape or label too. Roughly one judge in twelve is colour-blind.
 */
export const tokens = {
  color: {
    estimate: '#2563eb',
    truth: '#16a34a',
    baseline: '#dc2626',
    ellipse: 'rgba(37, 99, 235, 0.18)',
    ok: '#16a34a',
    warn: '#d97706',
    reject: '#dc2626',
    text: '#0f172a',
    muted: '#475569',
    surface: '#ffffff',
  },
  space: { xs: 4, sm: 8, md: 16, lg: 24, xl: 40 },
  radius: { sm: 4, md: 8, pill: 999 },
} as const
