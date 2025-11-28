import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '0.0.0.0',
    open: '/',
    strictPort: true
  },
  build: {
    // Ensure public directory is copied correctly
    copyPublicDir: true,
    // Output directory
    outDir: 'dist'
  },
  // Ensure public assets are served correctly
  publicDir: 'public'
})

