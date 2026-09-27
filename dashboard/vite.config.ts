import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// Бэкенд для live-режима в dev: VITE_API_TARGET (по умолчанию локальный api на 8000), Grafana — VITE_GRAFANA_TARGET.
const apiTarget = process.env.VITE_API_TARGET ?? 'http://localhost:8000'
const grafanaTarget = process.env.VITE_GRAFANA_TARGET ?? 'http://localhost:3000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: apiTarget, changeOrigin: true },
      '/health': { target: apiTarget, changeOrigin: true },
      '/ws': { target: apiTarget, ws: true, changeOrigin: true },
      '/grafana': { target: grafanaTarget, changeOrigin: true, ws: true },
    },
  },
  worker: {
    format: 'es',
  },
  build: {
    target: 'es2023',
    sourcemap: false,
    chunkSizeWarningLimit: 1500,
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            { name: 'maplibre', test: /node_modules[\\/](maplibre-gl|@maplibre|@mapbox)[\\/]/, priority: 40 },
            { name: 'echarts', test: /node_modules[\\/](echarts|zrender)[\\/]/, priority: 40 },
            {
              name: 'msw',
              test: /node_modules[\\/](msw|@mswjs|@open-draft|graphql|headers-polyfill|outvariant|strict-event-emitter|path-to-regexp|cookie|tough-cookie|rettime|statuses|until-async|is-node-process)[\\/]/,
              priority: 35,
            },
            {
              name: 'antd',
              test: /node_modules[\\/](antd|@ant-design|@rc-component|rc-[^\\/]+|@emotion|dayjs|stylis)[\\/]/,
              priority: 30,
            },
            {
              name: 'react',
              test: /node_modules[\\/](react|react-dom|scheduler|react-router|@tanstack)[\\/]/,
              priority: 20,
            },
          ],
        },
      },
    },
  },
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.{ts,tsx}'],
    restoreMocks: true,
  },
})
