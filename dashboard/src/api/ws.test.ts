import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  backoffDelay,
  DEFAULT_BACKOFF,
  ReconnectingSocket,
  useWebSocket,
  type SocketLike,
  type WsState,
} from './ws'

/** Управляемый сокет: тест сам открывает, закрывает и шлёт сообщения. */
class FakeSocket implements SocketLike {
  readyState = 0
  onopen: ((ev: Event) => void) | null = null
  onclose: ((ev: CloseEvent) => void) | null = null
  onerror: ((ev: Event) => void) | null = null
  onmessage: ((ev: MessageEvent) => void) | null = null
  closedWith: number | null = null
  readonly url: string

  constructor(url: string) {
    this.url = url
  }

  serverOpen(): void {
    this.readyState = 1
    this.onopen?.(new Event('open'))
  }

  serverSend(data: unknown): void {
    this.onmessage?.(
      new MessageEvent('message', { data: typeof data === 'string' ? data : JSON.stringify(data) }),
    )
  }

  serverDrop(): void {
    this.readyState = 3
    this.onerror?.(new Event('error'))
    this.onclose?.(new CloseEvent('close', { code: 1006 }))
  }

  close(code = 1000): void {
    this.closedWith = code
    this.readyState = 3
  }
}

function harness(options: { silenceTimeoutMs?: number } = {}) {
  const sockets: FakeSocket[] = []
  const messages: unknown[] = []
  const states: WsState[] = []
  const onOpen = vi.fn()
  const rs = new ReconnectingSocket(
    {
      url: 'ws://test/ws',
      createSocket: (url) => {
        const s = new FakeSocket(url)
        sockets.push(s)
        return s
      },
      onMessage: (d) => messages.push(d),
      onOpen,
      random: () => 0.5,
      silenceTimeoutMs: options.silenceTimeoutMs ?? 0,
    },
    (s) => states.push(s),
  )
  return { rs, sockets, messages, states, onOpen, last: () => states[states.length - 1] }
}

describe('backoffDelay', () => {
  it('растёт экспоненциально до потолка', () => {
    expect([1, 2, 3, 4, 5, 6].map((a) => backoffDelay(a))).toEqual([1000, 2000, 4000, 8000, 15000, 15000])
  })

  it('джиттер ±20% и потолок с учётом джиттера', () => {
    expect(backoffDelay(1, DEFAULT_BACKOFF, 0)).toBe(800)
    expect(backoffDelay(1, DEFAULT_BACKOFF, 1)).toBe(1200)
    expect(backoffDelay(10, DEFAULT_BACKOFF, 1)).toBe(15000)
  })
})

describe('ReconnectingSocket', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('подключается, разбирает JSON и сообщает об открытии', () => {
    const h = harness()
    h.rs.start()
    expect(h.sockets).toHaveLength(1)
    expect(h.last().status).toBe('connecting')
    h.sockets[0].serverOpen()
    expect(h.last()).toMatchObject({ status: 'open', attempt: 0 })
    expect(h.onOpen).toHaveBeenCalledTimes(1)
    h.sockets[0].serverSend({ type: 'clock', epoch: 1 })
    h.sockets[0].serverSend('not json')
    expect(h.messages).toEqual([{ type: 'clock', epoch: 1 }])
  })

  it('переподключается с растущей задержкой и сбрасывает счётчик после успеха', () => {
    const h = harness()
    h.rs.start()
    h.sockets[0].serverOpen()
    h.sockets[0].serverDrop()
    expect(h.last()).toMatchObject({ status: 'reconnecting', attempt: 1 })
    expect(h.last().nextRetryAt).toBe(Date.now() + 1000)

    vi.advanceTimersByTime(999)
    expect(h.sockets).toHaveLength(1)
    vi.advanceTimersByTime(1)
    expect(h.sockets).toHaveLength(2)

    // вторая попытка тоже неудачна — ждём уже 2 с
    h.sockets[1].serverDrop()
    expect(h.last()).toMatchObject({ status: 'reconnecting', attempt: 2 })
    vi.advanceTimersByTime(2000)
    expect(h.sockets).toHaveLength(3)

    h.sockets[2].serverOpen()
    expect(h.last()).toMatchObject({ status: 'open', attempt: 0 })
    expect(h.onOpen).toHaveBeenCalledTimes(2)

    // после успешного соединения задержка снова 1 с
    h.sockets[2].serverDrop()
    expect(h.last().attempt).toBe(1)
    vi.advanceTimersByTime(1000)
    expect(h.sockets).toHaveLength(4)
  })

  it('рвёт «зависшее» соединение по тишине', () => {
    const h = harness({ silenceTimeoutMs: 5000 })
    h.rs.start()
    h.sockets[0].serverOpen()
    vi.advanceTimersByTime(3000)
    h.sockets[0].serverSend({ type: 'clock' })
    vi.advanceTimersByTime(4000)
    expect(h.sockets[0].closedWith).toBeNull()
    vi.advanceTimersByTime(2000)
    expect(h.sockets[0].closedWith).toBe(4000)
    expect(h.last().status).toBe('reconnecting')
    vi.advanceTimersByTime(1000)
    expect(h.sockets).toHaveLength(2)
    h.rs.stop()
  })

  it('stop: закрывает сокет и больше не переподключается', () => {
    const h = harness()
    h.rs.start()
    h.sockets[0].serverOpen()
    h.rs.stop()
    expect(h.sockets[0].closedWith).toBe(1000)
    expect(h.last().status).toBe('closed')
    // событие закрытия от сервера после stop игнорируется
    h.sockets[0].serverDrop()
    vi.advanceTimersByTime(60_000)
    expect(h.sockets).toHaveLength(1)
  })

  it('ошибка создания сокета — тоже повод для переподключения', () => {
    let calls = 0
    const states: WsState[] = []
    const rs = new ReconnectingSocket(
      {
        url: 'ws://x',
        createSocket: (url) => {
          calls += 1
          if (calls === 1) throw new Error('bad url')
          return new FakeSocket(url)
        },
        onMessage: () => undefined,
        random: () => 0.5,
      },
      (s) => states.push(s),
    )
    rs.start()
    expect(states[states.length - 1]).toMatchObject({ status: 'reconnecting', attempt: 1 })
    vi.advanceTimersByTime(1000)
    expect(calls).toBe(2)
    rs.stop()
  })
})

describe('useWebSocket', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('отдаёт статус, сообщения и переподключается после обрыва', () => {
    const sockets: FakeSocket[] = []
    const createSocket = (url: string) => {
      const s = new FakeSocket(url)
      sockets.push(s)
      return s
    }
    const onMessage = vi.fn()
    const { result, unmount } = renderHook(() =>
      useWebSocket({ url: 'ws://test/ws', onMessage, createSocket, silenceTimeoutMs: 0 }),
    )
    expect(result.current.status).toBe('connecting')
    act(() => sockets[0].serverOpen())
    expect(result.current.status).toBe('open')
    act(() => sockets[0].serverSend({ type: 'delta', vehicles: [] }))
    expect(onMessage).toHaveBeenCalledWith({ type: 'delta', vehicles: [] })

    act(() => sockets[0].serverDrop())
    expect(result.current.status).toBe('reconnecting')
    expect(result.current.attempt).toBe(1)
    act(() => {
      vi.advanceTimersByTime(1500)
    })
    expect(sockets).toHaveLength(2)
    act(() => sockets[1].serverOpen())
    expect(result.current).toMatchObject({ status: 'open', attempt: 0 })

    unmount()
    expect(sockets[1].closedWith).toBe(1000)
    vi.advanceTimersByTime(60_000)
    expect(sockets).toHaveLength(2)
  })

  it('enabled: false — не подключается', () => {
    const createSocket = vi.fn((url: string) => new FakeSocket(url))
    const { result } = renderHook(() =>
      useWebSocket({ url: 'ws://x', onMessage: () => undefined, createSocket, enabled: false }),
    )
    expect(createSocket).not.toHaveBeenCalled()
    expect(result.current.status).toBe('closed')
  })
})
