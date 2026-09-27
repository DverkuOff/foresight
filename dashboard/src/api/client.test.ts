import { describe, expect, it } from 'vitest'
import { detailText } from './client'

describe('detailText', () => {
  it('строка — как есть, ошибки валидации — их сообщения', () => {
    expect(detailText('route R99 not found')).toBe('route R99 not found')
    expect(
      detailText([
        {
          type: 'value_error',
          loc: ['body', 'risk'],
          msg: 'Value error, green thresholds must not exceed the red ones',
        },
        { type: 'missing', loc: ['body', 'x'], msg: 'Field required' },
      ]),
    ).toBe('green thresholds must not exceed the red ones; Field required')
    expect(detailText({ a: 1 })).toBe('{"a":1}')
  })
})
