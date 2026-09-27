/**
 * Запуск mock-режима: симулятор из фикстур `dataset/test`, REST через обработчики MSW (`getResponse` — без service
 * worker: он требует https или localhost, а демо открывают и по http с адреса сервера), WebSocket — {@link MockSocket}.
 *
 * Демонстрация отказов (`?chaos=`): `redis` — деградация (Redis «недоступен», данные из кэша),
 * `offline` — через 5 с пропадает связь с сервером (REST и WebSocket).
 */
import { getResponse } from 'msw'
import { setTransport } from '../api/client'
import type { SocketFactory } from '../api/ws'
import type { HonestyFixture, WorldFixture } from './fixtures'
import honestyJson from './fixtures/honesty.json'
import worldJson from './fixtures/world.json'
import { createHandlers } from './handlers'
import { Simulator } from './simulator'
import { MockSocket } from './socket'

const EMIT_MS = 500
const OFFLINE_AFTER_MS = 5000

export type MockChaos = 'none' | 'redis' | 'offline'

export function parseChaos(value: string | null): MockChaos {
  return value === 'redis' || value === 'offline' ? value : 'none'
}

export interface MockRuntime {
  sim: Simulator
  createSocket: SocketFactory
  stop: () => void
}

export function createMockRuntime(speed: number, fleet = 0, chaos: MockChaos = 'none'): MockRuntime {
  const sim = new Simulator(worldJson as unknown as WorldFixture, honestyJson as unknown as HonestyFixture, {
    fleet,
  })
  sim.degraded = chaos === 'redis'
  let online = true
  const sockets = new Set<MockSocket>()
  const offlineTimer =
    chaos === 'offline'
      ? setTimeout(() => {
          online = false
          sockets.forEach((s) => s.drop())
          sockets.clear()
        }, OFFLINE_AFTER_MS)
      : null

  const handlers = createHandlers(sim)
  setTransport(async (request) => {
    await new Promise((r) => setTimeout(r, 40 + Math.random() * 80))
    if (!online) throw new TypeError('Failed to fetch (mock offline)')
    const response = await getResponse(handlers, request)
    return response ?? new Response(JSON.stringify({ detail: 'not found (mock)' }), { status: 404 })
  })
  let n = 0
  const timer = setInterval(() => {
    sim.advance(EMIT_MS * speed)
    n += 1
    // часы — раз в секунду, ТС — дельтой каждые 0,5 с, полный снимок — раз в 15 с
    sim.broadcast(n % 30 === 0 ? sim.vehicleMessage('snapshot') : sim.vehicleMessage('delta'))
    if (n % 2 === 0)
      sim.broadcast({ type: 'clock', stream_time: new Date(sim.now).toISOString(), epoch: sim.epoch })
  }, EMIT_MS)
  return {
    sim,
    createSocket: () => {
      const socket = new MockSocket(sim, () => online)
      if (online) sockets.add(socket)
      return socket
    },
    stop: () => {
      clearInterval(timer)
      if (offlineTimer) clearTimeout(offlineTimer)
    },
  }
}
