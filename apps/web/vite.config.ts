import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  server: {
    // Bound to every interface so a phone on the hotspot can open the UI too.
    host: true,
    proxy: {
      // Tiles and sockets come from the gateway, never from the internet. Keeping
      // the proxy here means the production build uses plain relative paths and
      // cannot accidentally acquire an external host. See docs/DEMO_RUNBOOK.md.
      '/tiles': 'http://127.0.0.1:8000',
      '/reports': 'http://127.0.0.1:8000',
      '/live': { target: 'ws://127.0.0.1:8000', ws: true },
      '/control': 'http://127.0.0.1:8000',
    },
  },
  build: { outDir: 'dist', sourcemap: true },
})
