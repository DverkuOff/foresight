/**
 * Подложка карты. По умолчанию — векторный светлый стиль OpenFreeMap (данные OpenStreetMap, без ключей),
 * перекрашенный в палитру интерфейса, с русскими названиями и без иконок POI. Стиль подгружается при старте
 * карты; не ответил за несколько секунд (нет интернета) — растровые тайлы из `config.tiles`, а без них карта
 * работает на фоне: линии маршрутов и ТС видны всегда.
 */
import type { LayerSpecification, Map as MapLibreMap, StyleSpecification } from 'maplibre-gl'
import { config } from '../config'

/** Цвета подложки в тон интерфейсу (светлая, серо-голубая, приглушённая — не спорит с цветами риска). */
export const BASE = {
  background: '#eef2f6',
  water: '#cfe2f2',
  residential: '#e7ecf1',
  green: '#dfeadf',
  building: '#dde3ea',
  roadMinor: '#ffffff',
  roadMajor: '#ffffff',
  roadCasing: 'rgba(148, 163, 184, 0.6)',
  motorway: '#fbfcfd',
  rail: '#c5cdd6',
  boundary: '#b4bfcb',
  label: '#475569',
  labelMinor: '#7b8794',
  waterLabel: '#4a7fb0',
  halo: '#ffffff',
} as const

const RU_NAME = ['coalesce', ['get', 'name:ru'], ['get', 'name']]

function colorFor(id: string): string | null {
  if (id === 'background') return BASE.background
  if (id === 'water' || id === 'waterway') return BASE.water
  if (id.startsWith('landuse_residential')) return BASE.residential
  if (id.includes('wood') || id.includes('park') || id.includes('grass')) return BASE.green
  if (id === 'building') return BASE.building
  if (id.includes('dashline')) return BASE.background
  if (id.startsWith('railway')) return BASE.rail
  if (id.startsWith('boundary')) return BASE.boundary
  if (id.includes('casing')) return BASE.roadCasing
  if (id.startsWith('highway_motorway')) return id.endsWith('subtle') ? BASE.roadMinor : BASE.motorway
  if (id.startsWith('highway_major')) return id.endsWith('subtle') ? BASE.roadMinor : BASE.roadMajor
  if (id.startsWith('highway_minor') || id.startsWith('highway_path') || id.startsWith('aeroway')) {
    return BASE.roadMinor
  }
  return null
}

/** Крупные населённые пункты — темнее (у стилей OpenFreeMap слои `place_*` или `label_*`). */
const MAJOR_PLACE = /^(place|label)_(city|town|suburb|state)/

function labelColor(id: string): string {
  if (id.startsWith('water')) return BASE.waterLabel
  return MAJOR_PLACE.test(id) ? BASE.label : BASE.labelMinor
}

/** Перекрасить стиль OpenFreeMap: палитра интерфейса, русские названия, без иконок (спрайты не нужны). */
export function tintStyle(style: StyleSpecification): StyleSpecification {
  const layers: LayerSpecification[] = []
  for (const layer of style.layers ?? []) {
    // правка как простого объекта: типы слоёв MapLibre не допускают запись произвольных свойств
    const l = structuredClone(layer) as unknown as {
      id: string
      type: string
      layout?: Record<string, unknown>
      paint?: Record<string, unknown>
    }
    if (l.layout && 'icon-image' in l.layout) continue // POI, щиты дорог, стрелки одностороннего движения
    const color = colorFor(l.id)
    if (color && l.paint) {
      if (l.type === 'background') l.paint['background-color'] = color
      if (l.type === 'fill') l.paint['fill-color'] = color
      if (l.type === 'line') l.paint['line-color'] = color
    }
    if (l.type === 'symbol' && l.layout) {
      const field = JSON.stringify(l.layout['text-field'] ?? '')
      if (field.includes('name')) l.layout['text-field'] = RU_NAME
      l.paint = { ...(l.paint ?? {}), 'text-color': labelColor(l.id), 'text-halo-color': BASE.halo }
    }
    layers.push(l as unknown as LayerSpecification)
  }
  const out: StyleSpecification = { ...style, layers }
  delete out.sprite
  return out
}

/** Загрузить и перекрасить векторный стиль (``null`` — нет адреса, нет ответа за ``timeoutMs`` или ошибка). */
export async function fetchBaseStyle(
  url: string | null,
  timeoutMs = 4000,
): Promise<StyleSpecification | null> {
  if (!url) return null
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), timeoutMs)
  try {
    const res = await fetch(url, { signal: ctrl.signal })
    if (!res.ok) return null
    const style = (await res.json()) as StyleSpecification
    return Array.isArray(style.layers) && style.sources ? tintStyle(style) : null
  } catch {
    return null
  } finally {
    clearTimeout(timer)
  }
}

/** Слои подложки под слоями приложения (``beforeId`` — нижний слой приложения). */
export function addBaseStyle(map: MapLibreMap, style: StyleSpecification, beforeId: string): void {
  if (style.glyphs) map.setGlyphs(style.glyphs)
  for (const [id, source] of Object.entries(style.sources)) {
    if (!map.getSource(id)) map.addSource(id, source)
  }
  for (const layer of style.layers) {
    if (layer.type === 'background') {
      map.setPaintProperty('background', 'background-color', BASE.background)
      continue
    }
    map.addLayer(layer, beforeId)
  }
}

/** Растровая подложка из `config.tiles` (светлые тайлы OSM приглушаются, тёмные — инвертируются). */
export function addRasterBase(map: MapLibreMap, beforeId: string): void {
  if (!config.tiles.length) return
  map.addSource('basemap', {
    type: 'raster',
    tiles: config.tiles,
    tileSize: 256,
    maxzoom: 19,
    attribution: config.tilesAttribution,
  })
  map.addLayer(
    {
      id: 'basemap',
      type: 'raster',
      source: 'basemap',
      // светлые тайлы OSM — как есть, но приглушённые (не спорят с цветами риска); тёмные (CARTO dark) —
      // инверсия яркости (min > max) и поворот тона возвращают светлую подложку
      paint: config.tilesInvert
        ? { 'raster-opacity': 0.9, 'raster-saturation': -0.55, 'raster-fade-duration': 150 }
        : {
            'raster-opacity': 0.9,
            'raster-brightness-min': 1,
            'raster-brightness-max': 0.15,
            'raster-hue-rotate': 180,
            'raster-saturation': -0.55,
            'raster-fade-duration': 150,
          },
    },
    beforeId,
  )
}
