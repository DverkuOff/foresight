import { describe, expect, it } from 'vitest'
import { DEFAULT_BASEMAP_STYLE, DEFAULT_TILES, resolveConfig } from './config'

describe('resolveConfig', () => {
  it('по умолчанию: dev → mock, production → live', () => {
    expect(resolveConfig('', {}, { DEV: true }).apiMode).toBe('mock')
    expect(resolveConfig('', {}, { DEV: false }).apiMode).toBe('live')
  })

  it('приоритет: адресная строка → runtime-конфиг → VITE_API_MODE', () => {
    expect(
      resolveConfig('?mode=live', { apiMode: 'mock' }, { VITE_API_MODE: 'mock', DEV: true }).apiMode,
    ).toBe('live')
    expect(resolveConfig('', { apiMode: 'mock' }, { VITE_API_MODE: 'live' }).apiMode).toBe('mock')
    expect(resolveConfig('', { apiMode: 'bogus' }, { VITE_API_MODE: 'live', DEV: true }).apiMode).toBe('live')
  })

  it('подложка: OSM по умолчанию (затемняется на карте) с атрибуцией, `none` — без подложки', () => {
    const def = resolveConfig('', {}, {})
    expect(def.tiles).toEqual([DEFAULT_TILES])
    expect(def.tilesInvert).toBe(true)
    expect(def.tilesAttribution).toContain('OpenStreetMap')
    const carto = resolveConfig(
      '',
      { tilesUrl: 'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png' },
      {},
    )
    expect(carto.tilesInvert).toBe(false)
    expect(carto.tilesAttribution).toContain('CARTO')
    expect(resolveConfig('', { tilesUrl: 'none' }, {}).tiles).toEqual([])
    // векторный стиль: по умолчанию OpenFreeMap; явные тайлы (в т. ч. none) его отключают; none — отключить
    expect(def.basemapStyle).toBe(DEFAULT_BASEMAP_STYLE)
    expect(resolveConfig('', { tilesUrl: 'none' }, {}).basemapStyle).toBeNull()
    expect(resolveConfig('', { basemapStyle: 'none' }, {}).basemapStyle).toBeNull()
    expect(
      resolveConfig('', { basemapStyle: 'https://x/style.json', tilesUrl: 'none' }, {}).basemapStyle,
    ).toBe('https://x/style.json')
    const osm = resolveConfig('', { tilesUrl: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png' }, {})
    expect(osm.tilesAttribution).not.toContain('CARTO')
  })

  it('Grafana без хвостового слэша, скорость mock ограничена', () => {
    expect(resolveConfig('', { grafanaUrl: '/grafana/' }, {}).grafanaUrl).toBe('/grafana')
    expect(resolveConfig('', {}, {}).grafanaUrl).toBe('/grafana')
    expect(resolveConfig('?speed=1000', {}, {}).mockSpeed).toBe(120)
    expect(resolveConfig('?speed=-3', {}, {}).mockSpeed).toBe(6)
    expect(resolveConfig('', { mockSpeed: '12' }, {}).mockSpeed).toBe(12)
  })
})
