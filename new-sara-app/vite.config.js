import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // API requests to the FastAPI server
      '/v1': {
        target: 'http://localhost:7009',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:7009',
        changeOrigin: true,
      },
      '/api': {
        target: 'http://localhost:7009',
        changeOrigin: true,
      },
    },
    historyApiFallback: true
    }
})