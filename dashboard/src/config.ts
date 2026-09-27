/**
 * Конфигурация дашборда. Источники по убыванию приоритета:
 * 1. параметр адресной строки `?mode=mock|live` (удобно для демо);
 * 2. runtime-конфиг `window.__FORESIGHT_CONFIG__` из `/config.js` (в Docker пишется из переменных окружения);
 * 3. переменные сборки Vite (`VITE_API_MODE`, `VITE_TILES_URL`, `VITE_GRAFANA_URL`);
 * 4. значения по умолчанию: dev → mock, production → live.
 */

export type ApiMode = 'mock' | 'live'

interface RuntimeConfig {
  apiMode?: string
  tilesUrl?: string
  basemapStyle?: string
  grafanaUrl?: string
  mockSpeed?: number | string
}

declare global {
  interface Window {
    __FORESIGHT_CONFIG__?: RuntimeConfig
  }
}

/**
 * Запасная подложка — растровые тайлы OpenStreetMap без ключа (на карте приглушаются фильтром слоя). Тайлам OSM
 * нужен Referer (nginx отдаёт Referrer-Policy с origin).
 * Тёмные CARTO (`…/dark_all/…`) без ключа API рисуют водяной знак — только со своим ключом в FORESIGHT_TILES_URL.
 */
export const DEFAULT_TILES = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'

/**
 * Векторная подложка по умолчанию — светлый стиль OpenFreeMap positron (без ключей, те же слои, что у тёмного),
 * перекрашенный в палитру интерфейса (`components/basemap.ts`). Не загрузился — растровые `tiles`. `none` — без
 * векторного стиля.
 */
export const DEFAULT_BASEMAP_STYLE = 'https://tiles.openfreemap.org/styles/positron'

const OSM_ATTRIBUTION =
  '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors'

export interface AppConfig {
  apiMode: ApiMode
  /** Шаблоны URL растровых тайлов (через `|`), `none` — без подложки. */
  tiles: string[]
  /** URL векторного стиля подложки (`null` — только растровые тайлы). */
  basemapStyle: string | null
  /** Тайлы светлые (OSM) — только приглушать; тёмные (например, CARTO dark) — инвертировать в светлые. */
  tilesInvert: boolean
  tilesAttribution: string
  grafanaUrl: string
  /** Ускорение времени потока в mock-режиме. */
  mockSpeed: number
}

function asMode(value: string | undefined | null): ApiMode | null {
  return value === 'mock' || value === 'live' ? value : null
}

function nonEmpty(value: string | undefined | null): string | null {
  return value && value.trim() !== '' ? value.trim() : null
}

export function resolveConfig(
  search: string,
  runtime: RuntimeConfig,
  env: Record<string, string | boolean | undefined>,
): AppConfig {
  const params = new URLSearchParams(search)
  const envMode = typeof env.VITE_API_MODE === 'string' ? env.VITE_API_MODE : undefined
  const apiMode =
    asMode(params.get('mode')) ?? asMode(runtime.apiMode) ?? asMode(envMode) ?? (env.DEV ? 'mock' : 'live')
  const envTiles = typeof env.VITE_TILES_URL === 'string' ? env.VITE_TILES_URL : undefined
  const tilesExplicit = nonEmpty(runtime.tilesUrl) ?? nonEmpty(envTiles)
  const tilesRaw = tilesExplicit ?? DEFAULT_TILES
  const envStyle = typeof env.VITE_BASEMAP_STYLE === 'string' ? env.VITE_BASEMAP_STYLE : undefined
  // явно заданные тайлы (в т. ч. `none` — стенд без интернета) важнее векторного стиля по умолчанию
  const styleRaw =
    nonEmpty(runtime.basemapStyle) ?? nonEmpty(envStyle) ?? (tilesExplicit ? null : DEFAULT_BASEMAP_STYLE)
  const tiles = tilesRaw === 'none' ? [] : tilesRaw.split('|').filter((t) => t.trim() !== '')
  const isCarto = tiles.some((t) => t.includes('cartocdn'))
  const envGrafana = typeof env.VITE_GRAFANA_URL === 'string' ? env.VITE_GRAFANA_URL : undefined
  const grafanaUrl = (nonEmpty(runtime.grafanaUrl) ?? nonEmpty(envGrafana) ?? '/grafana').replace(/\/+$/, '')
  const speedRaw = Number(params.get('speed') ?? runtime.mockSpeed ?? 6)
  const mockSpeed = Number.isFinite(speedRaw) && speedRaw > 0 ? Math.min(speedRaw, 120) : 6
  return {
    apiMode,
    tiles,
    basemapStyle: styleRaw && styleRaw !== 'none' ? styleRaw : null,
    tilesInvert: !tiles.some((t) => /dark/i.test(t)),
    tilesAttribution: isCarto
      ? `${OSM_ATTRIBUTION}, © <a href="https://carto.com/attributions" target="_blank" rel="noreferrer">CARTO</a>`
      : OSM_ATTRIBUTION,
    grafanaUrl,
    mockSpeed,
  }
}

export const config: AppConfig = resolveConfig(
  typeof window !== 'undefined' ? window.location.search : '',
  (typeof window !== 'undefined' ? window.__FORESIGHT_CONFIG__ : undefined) ?? {},
  import.meta.env,
)

export function wsUrl(path = '/ws'): string {
  const { protocol, host } = window.location
  return `${protocol === 'https:' ? 'wss' : 'ws'}://${host}${path}`
}
