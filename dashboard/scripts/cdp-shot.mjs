#!/usr/bin/env node
/**
 * Скриншоты страниц через Chrome DevTools Protocol — без зависимостей (Node 22+: глобальные fetch и WebSocket).
 * В отличие от `chrome --screenshot` ждёт в реальном времени (карта, WebSocket, анимации) и печатает ошибки страницы.
 *
 *   node cdp-shot.mjs --cdp http://127.0.0.1:19222 --base http://127.0.0.1:15180 --out /out \
 *     --sizes 1920x1080,1366x768 --wait 9000 overview=/ incident=/?incident=top
 *
 * Файлы: <out>/dashboard-<имя>-<ширина>x<высота>.png. Код выхода 1, если на странице были необработанные ошибки.
 */
import { writeFile } from 'node:fs/promises'

function parseArgs(argv) {
  const opts = {
    cdp: 'http://127.0.0.1:9222',
    base: '',
    out: '.',
    sizes: '1920x1080',
    wait: '8000',
    pages: [],
  }
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i]
    if (a.startsWith('--')) opts[a.slice(2)] = argv[++i]
    else {
      const eq = a.indexOf('=')
      opts.pages.push([a.slice(0, eq), a.slice(eq + 1)])
    }
  }
  return opts
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

/** Минимальный клиент CDP поверх WebSocket страницы. */
function connect(url) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url)
    let seq = 0
    const pending = new Map()
    const handlers = new Map()
    ws.addEventListener('message', (event) => {
      const msg = JSON.parse(String(event.data))
      if (msg.id && pending.has(msg.id)) {
        const { res, rej } = pending.get(msg.id)
        pending.delete(msg.id)
        if (msg.error) rej(new Error(`${msg.error.message} (${msg.error.code})`))
        else res(msg.result)
      } else if (msg.method) {
        for (const h of handlers.get(msg.method) ?? []) h(msg.params)
      }
    })
    ws.addEventListener('error', () => reject(new Error(`CDP connection failed: ${url}`)))
    ws.addEventListener('open', () =>
      resolve({
        send(method, params = {}) {
          seq += 1
          const id = seq
          ws.send(JSON.stringify({ id, method, params }))
          return new Promise((res, rej) => pending.set(id, { res, rej }))
        },
        on(method, handler) {
          handlers.set(method, [...(handlers.get(method) ?? []), handler])
        },
        close() {
          ws.close()
        },
      }),
    )
  })
}

async function shoot(opts, name, path, width, height) {
  const target = await (await fetch(`${opts.cdp}/json/new?about:blank`, { method: 'PUT' })).json()
  const client = await connect(target.webSocketDebuggerUrl)
  const problems = []
  client.on('Runtime.exceptionThrown', (p) => {
    const d = p.exceptionDetails
    problems.push(`exception: ${d.exception?.description ?? d.text}`.slice(0, 600))
  })
  client.on('Runtime.consoleAPICalled', (p) => {
    if (p.type === 'error' || p.type === 'warning') {
      const text = p.args.map((a) => a.value ?? a.description ?? '').join(' ')
      problems.push(`console.${p.type}: ${text}`.slice(0, 400))
    }
  })
  client.on('Log.entryAdded', (p) => {
    if (p.entry.level === 'error' && !/favicon|tile|cartocdn/i.test(p.entry.url ?? '')) {
      problems.push(`log: ${p.entry.text} ${p.entry.url ?? ''}`.slice(0, 400))
    }
  })
  await client.send('Runtime.enable')
  await client.send('Log.enable')
  await client.send('Page.enable')
  await client.send('Emulation.setDeviceMetricsOverride', {
    width,
    height,
    deviceScaleFactor: 1,
    mobile: false,
  })
  await client.send('Page.navigate', { url: `${opts.base}${path}` })
  await sleep(Number(opts.wait))
  const { data } = await client.send('Page.captureScreenshot', {
    format: 'png',
    captureBeyondViewport: false,
  })
  const file = `${opts.out}/dashboard-${name}-${width}x${height}.png`
  await writeFile(file, Buffer.from(data, 'base64'))
  client.close()
  await fetch(`${opts.cdp}/json/close/${target.id}`).catch(() => undefined)
  console.log(`ok  ${file}${problems.length ? `  (${problems.length} problems)` : ''}`)
  for (const p of problems) console.log(`    ${p}`)
  return problems.filter((p) => p.startsWith('exception')).length
}

const opts = parseArgs(process.argv.slice(2))
let exceptions = 0
for (const size of opts.sizes.split(',')) {
  const [w, h] = size.split('x').map(Number)
  for (const [name, path] of opts.pages) exceptions += await shoot(opts, name, path, w, h)
}
process.exit(exceptions ? 1 : 0)
