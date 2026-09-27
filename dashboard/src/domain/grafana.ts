/** Встраивание панелей Grafana (дашборды из deploy/grafana/dashboards, sub-path `/grafana`). */

export interface GrafanaPanel {
  uid: string
  panelId: number
  title: string
}

export const GRAFANA_DASHBOARDS: { uid: string; title: string }[] = [
  { uid: 'foresight-overview', title: 'Обзор системы' },
  { uid: 'foresight-stream', title: 'Поток и прогнозы' },
  { uid: 'foresight-model', title: 'Модель' },
  { uid: 'foresight-ingest', title: 'Приём NDTP' },
  { uid: 'foresight-storage', title: 'Хранилище и деградация' },
]

/** Панели для страницы «Производительность» (id панелей — из JSON дашбордов). */
export const PERF_PANELS: GrafanaPanel[] = [
  { uid: 'foresight-overview', panelId: 14, title: 'Пропускная способность: приём → шина → predictor' },
  { uid: 'foresight-overview', panelId: 15, title: 'Задержка обработки события' },
  { uid: 'foresight-model', panelId: 7, title: 'Латентность инференса POST /predict' },
  { uid: 'foresight-overview', panelId: 16, title: 'Лаг consumer group' },
  { uid: 'foresight-model', panelId: 14, title: 'Онлайн-MAE против baseline' },
  { uid: 'foresight-storage', panelId: 2, title: 'Доступность зависимостей по сервисам' },
]

export interface PanelUrlOptions {
  from?: string
  to?: string
  refresh?: string
}

/** URL одной панели (`/d-solo`) в светлой теме, режим киоска. */
export function grafanaPanelUrl(base: string, panel: GrafanaPanel, options: PanelUrlOptions = {}): string {
  const params = new URLSearchParams({
    orgId: '1',
    panelId: String(panel.panelId),
    theme: 'light',
    from: options.from ?? 'now-15m',
    to: options.to ?? 'now',
    refresh: options.refresh ?? '5s',
  })
  return `${base.replace(/\/+$/, '')}/d-solo/${encodeURIComponent(panel.uid)}/${encodeURIComponent(panel.uid)}?${params.toString()}&kiosk`
}

/** Ссылка «Открыть в Grafana» на дашборд (или конкретную панель). */
export function grafanaDashboardUrl(base: string, uid: string, panelId?: number): string {
  const root = `${base.replace(/\/+$/, '')}/d/${encodeURIComponent(uid)}/${encodeURIComponent(uid)}?orgId=1`
  return panelId === undefined ? root : `${root}&viewPanel=${panelId}`
}
