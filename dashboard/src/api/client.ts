/**
 * HTTP-клиент API. В live-режиме запросы идут в `fetch` (тот же origin, nginx проксирует `/api` в `api:8000`),
 * в mock-режиме транспорт подменяется обработчиками MSW (`src/mocks`).
 */

export type Transport = (request: Request) => Promise<Response>

let transport: Transport = (request) => fetch(request)

/** Подменить транспорт (mock-режим, тесты). */
export function setTransport(next: Transport): void {
  transport = next
}

export class ApiError extends Error {
  readonly status: number
  readonly detail: string

  constructor(status: number, detail: string) {
    super(`HTTP ${status}: ${detail}`)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

export type QueryParams = Record<string, string | number | boolean | null | undefined>

export function buildUrl(path: string, params?: QueryParams): string {
  const base = typeof window !== 'undefined' ? window.location.origin : 'http://localhost'
  const url = new URL(path, base)
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value !== undefined && value !== null && value !== '') url.searchParams.set(key, String(value))
  }
  return url.toString()
}

/** Текст ошибки FastAPI: строка как есть, ошибки валидации — их сообщения через «; ». */
export function detailText(detail: unknown): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    const messages = detail
      .map((e) => (e && typeof e === 'object' && 'msg' in e ? String((e as { msg: unknown }).msg) : ''))
      .map((m) => m.replace(/^Value error,\s*/, ''))
      .filter(Boolean)
    if (messages.length) return messages.join('; ')
  }
  return JSON.stringify(detail)
}

async function parse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = response.statusText || 'ошибка запроса'
    try {
      const body: unknown = await response.json()
      if (body && typeof body === 'object' && 'detail' in body) {
        detail = detailText((body as { detail: unknown }).detail)
      }
    } catch {
      // тело не JSON — оставляем statusText
    }
    throw new ApiError(response.status, detail)
  }
  return (await response.json()) as T
}

export async function apiGet<T>(path: string, params?: QueryParams, signal?: AbortSignal): Promise<T> {
  const request = new Request(buildUrl(path, params), { headers: { Accept: 'application/json' }, signal })
  return parse<T>(await transport(request))
}

export async function apiPost<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const request = new Request(buildUrl(path), {
    method: 'POST',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  return parse<T>(await transport(request))
}

export async function apiPut<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const request = new Request(buildUrl(path), {
    method: 'PUT',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  return parse<T>(await transport(request))
}
