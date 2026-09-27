/** Mock-WebSocket: подписка на события симулятора с тем же интерфейсом, что и браузерный WebSocket. */
import type { SocketLike } from '../api/ws'
import type { Simulator } from './simulator'

export class MockSocket implements SocketLike {
  readyState = 0
  onopen: ((ev: Event) => void) | null = null
  onclose: ((ev: CloseEvent) => void) | null = null
  onerror: ((ev: Event) => void) | null = null
  onmessage: ((ev: MessageEvent) => void) | null = null
  private unsubscribe: (() => void) | null = null
  private readonly sim: Simulator
  private readonly isOnline: () => boolean

  /** `isOnline` — «сеть» mock-сервера: без неё соединение не устанавливается (демо «нет связи»). */
  constructor(sim: Simulator, isOnline: () => boolean = () => true) {
    this.sim = sim
    this.isOnline = isOnline
    setTimeout(() => (this.isOnline() ? this.open() : this.drop()), 30)
  }

  private send(data: unknown): void {
    this.onmessage?.(new MessageEvent('message', { data: JSON.stringify(data) }))
  }

  private open(): void {
    if (this.readyState !== 0) return
    this.readyState = 1
    this.onopen?.(new Event('open'))
    this.send({ type: 'clock', stream_time: new Date(this.sim.now).toISOString(), epoch: this.sim.epoch })
    this.send(this.sim.vehicleMessage('snapshot'))
    this.unsubscribe = this.sim.subscribe((msg) => this.send(msg))
  }

  /** Обрыв со стороны «сервера» (код 1006, как при потере сети). */
  drop(): void {
    if (this.readyState >= 2) return
    this.readyState = 3
    this.unsubscribe?.()
    this.unsubscribe = null
    this.onerror?.(new Event('error'))
    this.onclose?.(new CloseEvent('close', { code: 1006 }))
  }

  close(): void {
    if (this.readyState >= 2) return
    this.readyState = 3
    this.unsubscribe?.()
    this.unsubscribe = null
    this.onclose?.(new CloseEvent('close', { code: 1000 }))
  }
}
