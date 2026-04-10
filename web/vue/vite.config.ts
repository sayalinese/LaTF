import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import * as path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  },
  server: {
    port: 3000,
    allowedHosts: true,
    proxy: {
      '/api': {
        target: 'http://localhost:5555', // Flask 后端（本地）
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        cookieDomainRewrite: { '127.0.0.1': 'localhost', '*': '' }
      }
    }
  }
})
