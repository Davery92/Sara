import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: ['talk.avery.cloud'],
    proxy: {
      // Add proxy for talk.avery.cloud
      '/avery-api': {
        target: 'https://talk.avery.cloud',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/avery-api/, '')
      },
      // Maintain existing proxies if any
      '/v1': {
        target: 'http://localhost:7009',
        changeOrigin: true
      },
      '/health': {
        target: 'http://localhost:7009',
        changeOrigin: true
      },
      '/ws': {
        target: 'ws://localhost:7009',
        ws: true
      }
    }
  }
})