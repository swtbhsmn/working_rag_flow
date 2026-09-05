import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

declare const process: { env: Record<string, string | undefined> }

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': process.env.VITE_API_TARGET || 'http://127.0.0.1:8000',
      '/health': process.env.VITE_API_TARGET || 'http://127.0.0.1:8000',
    },
  },
})
