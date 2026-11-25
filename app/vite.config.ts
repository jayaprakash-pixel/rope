import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev: the UI on 127.0.0.1:3005 (Port Registry), /api proxied to `rope serve` on 127.0.0.1:4003
// (start it with --dev-origin http://127.0.0.1:3005). Build: dist/ is served by `rope serve --static app/dist`.
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 3005,
    strictPort: true,
    proxy: { '/api': { target: 'http://127.0.0.1:4003', changeOrigin: true } },
  },
  // no data: URIs (fonts stay real files): the app's CSP allows fonts from this server only
  build: { outDir: 'dist', chunkSizeWarningLimit: 900, assetsInlineLimit: 0 },
})
