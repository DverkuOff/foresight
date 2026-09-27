/**
 * WebSocket с автоматическим переподключением (экспоненциальная задержка с джиттером) и сторожем тишины:
 * сервер шлёт `clock` раз в секунду, поэтому долгая тишина означает «зависшее» соединение — рвём и переподключаемся.
 */
import { useEffect, useRef, useState } from 'react'

/** Минимальный интерфейс WebSocket, который нужен хуку (браузерный WebSocket и mock-сокет его реализуют). */
export interface SocketLike {
  readonly readyState: number
  onopen: ((ev: Event) => void) | null
  onclose: ((ev: CloseEvent) => void) | null
  onerror: ((ev: Event) => void) | null
  onmessage: ((ev: MessageEvent) => void) | null
  close(code?: number, reason?: string): void
}

export type SocketFactory = (url: string) => SocketLike

export type WsStatus = 'connecting' | 'open' | 'reconnecting' | 'closed'

export interface WsState {
  status: WsStatus
  /** Номер текущей попытки переподключения (0 — соединение установлено или первая попытка). */
  attempt: number
  /** Когда будет следующая попытка (мс, `Date.now()`), если ждём. */
  nextRetryAt: number | null
  /** Когда соединение последний раз открылось. */
  openedAt: number | null
}

export interface BackoffOptions {
  initialMs: number
  maxMs: number
  factor: number
  /** Доля случайного разброса (0.2 = ±20%). */
  jitter: number
}

export const DEFAULT_BACKOFF: BackoffOptions = { initialMs: 1000, maxMs: 15000, factor: 2, jitter: 0.2 }

/** Задержка перед попыткой `attempt` (1, 2, …). */
export function backoffDelay(attempt: number, options: BackoffOptions = DEFAULT_BACKOFF, rnd = 0.5): number {
  const base = Math.min(options.maxMs, options.initialMs * options.factor ** Math.max(0, attempt - 1))
  const spread = 1 + options.jitter * (rnd * 2 - 1)
  return Math.round(Math.min(options.maxMs, base * spread))
}

export interface ReconnectOptions {
  url: string
  createSocket: SocketFactory
  onMessage: (data: unknown) => void
  onOpen?: () => void
  backoff?: BackoffOptions
  /** Через сколько мс тишины считать соединение зависшим (0 — не следить). */
  silenceTimeoutMs?: number
  random?: () => number
  now?: () => number
}

const OPEN = 1

export class ReconnectingSocket {
  private socket: SocketLike | null = null
  private attempt = 0
  private retryTimer: ReturnType<typeof setTimeout> | null = null
  private watchdog: ReturnType<typeof setInterval> | null = null
  private lastMessageAt = 0
  private stopped = true
  private state: WsState = { status: 'connecting', attempt: 0, nextRetryAt: null, openedAt: null }
  private readonly options: ReconnectOptions
  private readonly listener: (state: WsState) => void

  constructor(options: ReconnectOptions, listener: (state: WsState) => void) {
    this.options = options
    this.listener = listener
  }

  private now(): number {
    return (this.options.now ?? Date.now)()
  }

  private emit(patch: Partial<WsState>): void {
    this.state = { ...this.state, ...patch }
    this.listener(this.state)
  }

  start(): void {
    if (!this.stopped) return
    this.stopped = false
    this.attempt = 0
    this.open()
    const silence = this.options.silenceTimeoutMs ?? 0
    if (silence > 0) {
      this.watchdog = setInterval(
        () => {
          const socket = this.socket
          if (socket && socket.readyState === OPEN && this.now() - this.lastMessageAt > silence) {
            this.drop(socket)
          }
        },
        Math.min(1000, silence),
      )
    }
  }

  stop(): void {
    this.stopped = true
    if (this.retryTimer) clearTimeout(this.retryTimer)
    if (this.watchdog) clearInterval(this.watchdog)
    this.retryTimer = null
    this.watchdog = null
    const socket = this.socket
    this.socket = null
    if (socket) {
      detach(socket)
      socket.close(1000, 'client stop')
    }
    this.emit({ status: 'closed', nextRetryAt: null })
  }

  private open(): void {
    this.emit({
      status: this.attempt === 0 ? 'connecting' : 'reconnecting',
      attempt: this.attempt,
      nextRetryAt: null,
    })
    let socket: SocketLike
    try {
      socket = this.options.createSocket(this.options.url)
    } catch {
      this.scheduleReconnect()
      return
    }
    this.socket = socket
    socket.onopen = () => {
      if (this.socket !== socket) return
      this.attempt = 0
      this.lastMessageAt = this.now()
      this.emit({ status: 'open', attempt: 0, nextRetryAt: null, openedAt: this.now() })
      this.options.onOpen?.()
    }
    socket.onmessage = (event: MessageEvent) => {
      if (this.socket !== socket) return
      this.lastMessageAt = this.now()
      let data: unknown = event.data
      if (typeof data === 'string') {
        try {
          data = JSON.parse(data)
        } catch {
          return
        }
      }
      this.options.onMessage(data)
    }
    socket.onerror = () => {
      // за ошибкой всегда следует close — переподключение там
    }
    socket.onclose = () => {
      if (this.socket !== socket) return
      this.socket = null
      detach(socket)
      if (!this.stopped) this.scheduleReconnect()
    }
  }

  /** Принудительно разорвать «зависшее» соединение и переподключиться. */
  private drop(socket: SocketLike): void {
    if (this.socket !== socket) return
    this.socket = null
    detach(socket)
    try {
      socket.close(4000, 'silence timeout')
    } catch {
      // уже закрыт
    }
    if (!this.stopped) this.scheduleReconnect()
  }

  private scheduleReconnect(): void {
    this.attempt += 1
    const delay = backoffDelay(this.attempt, this.options.backoff, (this.options.random ?? Math.random)())
    this.emit({ status: 'reconnecting', attempt: this.attempt, nextRetryAt: this.now() + delay })
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null
      if (!this.stopped) this.open()
    }, delay)
  }
}

function detach(socket: SocketLike): void {
  socket.onopen = null
  socket.onclose = null
  socket.onerror = null
  socket.onmessage = null
}

export const browserSocket: SocketFactory = (url) => new WebSocket(url)

export interface UseWebSocketOptions {
  url: string
  onMessage: (data: unknown) => void
  onOpen?: () => void
  createSocket?: SocketFactory
  enabled?: boolean
  backoff?: BackoffOptions
  silenceTimeoutMs?: number
}

const INITIAL_STATE: WsState = { status: 'connecting', attempt: 0, nextRetryAt: null, openedAt: null }

/** React-хук над {@link ReconnectingSocket}. Сообщения отдаются колбэком (без перерисовки на каждое). */
export function useWebSocket(options: UseWebSocketOptions): WsState {
  const { url, enabled = true, createSocket = browserSocket, backoff, silenceTimeoutMs = 45000 } = options
  const [state, setState] = useState<WsState>(INITIAL_STATE)
  const handlers = useRef({ onMessage: options.onMessage, onOpen: options.onOpen })

  useEffect(() => {
    handlers.current = { onMessage: options.onMessage, onOpen: options.onOpen }
  }, [options.onMessage, options.onOpen])

  useEffect(() => {
    if (!enabled) return undefined
    const socket = new ReconnectingSocket(
      {
        url,
        createSocket,
        backoff,
        silenceTimeoutMs,
        onMessage: (data) => handlers.current.onMessage(data),
        onOpen: () => handlers.current.onOpen?.(),
      },
      setState,
    )
    socket.start()
    return () => socket.stop()
  }, [url, enabled, createSocket, backoff, silenceTimeoutMs])

  return enabled ? state : { ...state, status: 'closed' }
}
