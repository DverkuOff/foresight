/** Детерминированный «шум» для моков: одинаковый ключ — одинаковое значение (стабильные прогнозы между тиками). */

/** FNV-1a → [0, 1). */
export function hash01(key: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < key.length; i += 1) {
    h ^= key.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return ((h >>> 0) % 1_000_003) / 1_000_003
}

/** Стандартное нормальное по ключу (Бокс — Мюллер). */
export function gauss(key: string): number {
  const u1 = Math.max(1e-9, hash01(`${key}#1`))
  const u2 = hash01(`${key}#2`)
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

/** Функция распределения стандартного нормального. */
export function normCdf(x: number): number {
  // Abramowitz — Stegun 7.1.26
  const t = 1 / (1 + 0.3275911 * Math.abs(x / Math.SQRT2))
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-(x * x) / 2)
  return x >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y)
}
