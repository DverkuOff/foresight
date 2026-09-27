import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { browserSocket, type SocketFactory } from './api/ws'
import { App } from './App'
import { config } from './config'
import './styles.css'

async function bootstrap(): Promise<void> {
  let createSocket: SocketFactory = browserSocket
  if (config.apiMode === 'mock') {
    // mock-слой (симулятор + фикстуры) — отдельный чанк, в live-режиме не загружается
    const { createMockRuntime, parseChaos } = await import('./mocks/install')
    const params = new URLSearchParams(window.location.search)
    // ?fleet=300 — нагрузочная проверка карты; ?chaos=redis|offline — демонстрация деградации и потери связи
    const fleet = Number(params.get('fleet') ?? 0) || 0
    createSocket = createMockRuntime(config.mockSpeed, fleet, parseChaos(params.get('chaos'))).createSocket
  }
  const root = document.getElementById('root')
  if (!root) throw new Error('#root not found')
  createRoot(root).render(
    <StrictMode>
      <App createSocket={createSocket} />
    </StrictMode>,
  )
}

void bootstrap()
