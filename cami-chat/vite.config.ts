import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/auto_session_stream": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/auto_session": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
})
