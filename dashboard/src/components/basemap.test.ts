import type { StyleSpecification } from 'maplibre-gl'
import { describe, expect, it } from 'vitest'
import { BASE, tintStyle } from './basemap'

const STYLE = {
  version: 8,
  sprite: 'https://tiles.example/sprites/ofm',
  glyphs: 'https://tiles.example/fonts/{fontstack}/{range}.pbf',
  sources: { openmaptiles: { type: 'vector', url: 'https://tiles.example/planet' } },
  layers: [
    { id: 'background', type: 'background', paint: { 'background-color': 'rgb(12,12,12)' } },
    {
      id: 'water',
      type: 'fill',
      source: 'openmaptiles',
      'source-layer': 'water',
      paint: { 'fill-color': '#111' },
    },
    {
      id: 'poi',
      type: 'symbol',
      source: 'openmaptiles',
      'source-layer': 'poi',
      layout: { 'icon-image': 'x' },
    },
    {
      id: 'place_city',
      type: 'symbol',
      source: 'openmaptiles',
      'source-layer': 'place',
      layout: { 'text-field': ['concat', ['get', 'name:latin'], '\n', ['get', 'name:nonlatin']] },
      paint: { 'text-color': 'rgb(101,101,101)' },
    },
  ],
} as unknown as StyleSpecification

describe('подложка', () => {
  it('перекрашивает стиль в палитру интерфейса, названия — по-русски, без иконок и спрайта', () => {
    const out = tintStyle(STYLE)
    expect(out.sprite).toBeUndefined()
    expect(out.layers.map((l) => l.id)).toEqual(['background', 'water', 'place_city'])
    const [bg, water, city] = out.layers as unknown as {
      paint: Record<string, unknown>
      layout?: Record<string, unknown>
    }[]
    expect(bg.paint['background-color']).toBe(BASE.background)
    expect(water.paint['fill-color']).toBe(BASE.water)
    expect(city.layout?.['text-field']).toEqual(['coalesce', ['get', 'name:ru'], ['get', 'name']])
    expect(city.paint['text-color']).toBe(BASE.label)
    expect(STYLE.layers).toHaveLength(4) // исходный стиль не меняется
  })
})
