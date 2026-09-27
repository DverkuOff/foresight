/** SVG-путь спарклайна: `null` и нечисловые значения рвут линию. */
export function sparkPath(
  values: readonly (number | null)[],
  width: number,
  height: number,
  max: number,
): string {
  const n = values.length
  if (n < 2 || max <= 0) return ''
  let d = ''
  let pen = false
  values.forEach((v, i) => {
    if (v === null || !Number.isFinite(v)) {
      pen = false
      return
    }
    const x = (i / (n - 1)) * width
    const y = height - (Math.max(0, v) / max) * (height - 2) - 1
    d += `${pen ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`
    pen = true
  })
  return d
}
