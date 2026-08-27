import { tokens } from './tokens'

/**
 * Injects `tokens` as CSS custom properties on `:root`, once, at app start.
 *
 * OWNER: Akshit  |  MILESTONE: M4
 *
 * `tokens.ts` is the single source of truth described in its own docstring. CSS
 * cannot `import` a TypeScript object directly, so this is the bridge -- every
 * stylesheet reads `var(--color-estimate)` etc. rather than a hand-copied hex value,
 * which is what keeps a second, driftable copy of the palette from ever existing.
 */
export function applyDesignTokens(): void {
  const root = document.documentElement.style
  for (const [key, value] of Object.entries(tokens.color)) {
    root.setProperty(`--color-${key}`, value)
  }
  for (const [key, value] of Object.entries(tokens.space)) {
    root.setProperty(`--space-${key}`, `${value}px`)
  }
  for (const [key, value] of Object.entries(tokens.radius)) {
    root.setProperty(`--radius-${key}`, `${value}px`)
  }
}
