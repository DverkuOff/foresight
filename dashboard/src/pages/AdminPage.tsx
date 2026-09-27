/**
 * «Администрирование» (Ant Design): источник потока и приём NDTP, пороги риска и алертов (применяются на лету),
 * версии моделей с переобучением на потоке и fallback, журнал с фильтрами и экспортом CSV, здоровье сервисов,
 * справочники ТС, маршрутов и остановок. Backend — `/api/admin/*`, `/api/replay/*`, `/api/ingest/stats`.
 */
import {
  ApiOutlined,
  CloudDownloadOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  FileSearchOutlined,
  HeartOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  SlidersOutlined,
  SyncOutlined,
} from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Alert,
  App as AntApp,
  Badge,
  Button,
  Descriptions,
  Form,
  Input,
  InputNumber,
  Popconfirm,
  Progress,
  Segmented,
  Select,
  Space,
  Table,
  Tabs,
  Tag,
  TimePicker,
  Tooltip,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import type { Dayjs } from 'dayjs'
import { useEffect, useMemo, useRef, useState } from 'react'
import { ApiError, apiGet, apiPost, apiPut, buildUrl } from '../api/client'
import { useRoutes } from '../api/queries'
import type {
  AdminSettings,
  AlertOut,
  FallbackInfo,
  IngestStats,
  JournalOut,
  ModelsOut,
  ModelVersion,
  PredictionOut,
  ReplayStatus,
  RetrainState,
  RouteOut,
  ServiceHealth,
  ServicesOut,
  UnitOut,
} from '../api/types'
import { RiskDot, RiskTag } from '../components/RiskDot'
import { formatClock, formatIn, formatNumber, formatPercent, formatSignedSeconds } from '../domain/format'
import { routeEnds } from '../domain/routes'
import { useLive } from '../state/liveStore'

/** Сообщения API, которые видит оператор, — по-русски (по началу текста). */
const API_MESSAGES: [RegExp, string][] = [
  [/^green thresholds must not exceed the red ones/, 'зелёные пороги не могут быть выше красных'],
  [/^journal unavailable/, 'журнал недоступен: нет связи с PostgreSQL'],
  [/^service is not configured/, 'сервис не настроен в этой установке'],
  [/^unavailable/, 'сервис не отвечает'],
  [/^retraining is already running/, 'переобучение уже идёт'],
  [/^no journal of closed forecasts/, 'нет журнала закрытых прогнозов (PostgreSQL)'],
  [/^model is not loaded/, 'модель ещё не загружена'],
  [/^(BundleError|FeaturesVersionError|OSError|ValueError)/, 'версия не найдена или не загружается'],
]

function errorText(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401 || error.status === 403) {
      return 'Нужен пароль администратора (пользователь admin): обновите страницу и войдите'
    }
    const detail = API_MESSAGES.find(([re]) => re.test(error.detail))?.[1] ?? error.detail
    return error.status === 422 ? `Не сохранено: ${detail}` : `Ошибка ${error.status}: ${detail}`
  }
  return error instanceof Error ? error.message : String(error)
}

/** Ошибка запроса вкладки: без данных — «не загрузилось», с прежними данными — «могли устареть». */
function QueryAlert({
  query,
  what,
}: {
  query: { isError: boolean; error: unknown; data?: unknown }
  what: string
}) {
  if (!query.isError) return null
  const stale = query.data !== undefined
  return (
    <Alert
      className="admin-alert"
      type={stale ? 'warning' : 'error'}
      showIcon
      message={stale ? `${what}: показаны последние полученные данные` : `${what}: не удалось загрузить`}
      description={errorText(query.error)}
    />
  )
}

function plural(n: number, one: string, few: string, many: string): string {
  const m10 = n % 10
  const m100 = n % 100
  if (m10 === 1 && m100 !== 11) return one
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few
  return many
}

const STATE_BADGE: Record<string, 'success' | 'error' | 'warning' | 'default' | 'processing'> = {
  up: 'success',
  running: 'processing',
  loading: 'processing',
  waiting: 'processing',
  down: 'error',
  failed: 'error',
  degraded: 'warning',
  unknown: 'default',
  disabled: 'default',
  idle: 'default',
  stopped: 'default',
  finished: 'default',
}

const STATE_TEXT: Record<string, string> = {
  up: 'работает',
  down: 'недоступен',
  degraded: 'деградация',
  unknown: 'нет данных',
  disabled: 'отключён',
  running: 'идёт',
  loading: 'загружает данные',
  waiting: 'ждёт приёмник',
  idle: 'ожидает',
  stopped: 'остановлен',
  finished: 'закончен',
  failed: 'ошибка',
}

const MODE_TEXT: Record<string, string> = {
  ndtp: 'replayer → приёмник NDTP напрямую',
  bridge: 'replayer → официальный эмулятор NDTP → приёмник',
}

// ---------------------------------------------------------------- поток

const SPEEDS = [1, 10, 30, 60]
const STARTS = ['06:00', '07:00', '08:00', '12:00', '17:00', '20:00']

function StreamTab() {
  const { message } = AntApp.useApp()
  const client = useQueryClient()
  // тот же ключ, что у шапки: один опрос статуса на страницу
  const status = useQuery({
    queryKey: ['replay-status'],
    queryFn: ({ signal }) => apiGet<ReplayStatus>('/api/replay/status', undefined, signal),
    refetchInterval: 2000,
    retry: 0,
  })
  const ingest = useQuery({
    queryKey: ['admin', 'ingest'],
    queryFn: ({ signal }) => apiGet<IngestStats>('/api/ingest/stats', undefined, signal),
    refetchInterval: 3000,
    retry: 0,
  })
  const [speedChoice, setSpeed] = useState<number | null>(null)
  const [startChoice, setStart] = useState<string | null>(null)
  const [modeChoice, setMode] = useState<string | null>(null)
  const s = status.data
  // элементы управления показывают то, что идёт сейчас, пока оператор не выбрал своё
  const speed = speedChoice ?? (s && SPEEDS.includes(s.speed) ? s.speed : 30)
  const start = startChoice ?? (s?.start && STARTS.includes(s.start) ? s.start : '06:00')
  const mode = modeChoice ?? (s && s.mode in MODE_TEXT ? s.mode : 'ndtp')
  const act = useMutation({
    mutationFn: ({ path, body }: { path: string; body?: unknown }) => apiPost<ReplayStatus>(path, body ?? {}),
    onSuccess: (_, v) => {
      void client.invalidateQueries({ queryKey: ['replay-status'] })
      message.success(
        v.path.endsWith('start')
          ? 'Поток запущен заново'
          : v.path.endsWith('stop')
            ? 'Поток остановлен'
            : 'Скорость изменена',
      )
    },
    onError: (e) => message.error(errorText(e)),
  })
  const running = s?.state === 'running'
  const i = ingest.data
  return (
    <div className="admin-tab">
      <p className="admin-tab__hint">
        Демо-поток: replayer проигрывает телеметрию дня из датасета как настоящий NDTP — сам или через
        официальный эмулятор. Перезапуск возвращает часы потока назад: приём и прогнозы начинают новую линию
        времени, открытые прогнозы закрываются как «сброс».
      </p>
      <QueryAlert query={status} what="Демо-поток (replayer)" />
      <Descriptions size="middle" column={{ xs: 1, md: 2, xl: 3 }} bordered title="Источник потока">
        <Descriptions.Item label="Состояние">
          <Badge
            status={STATE_BADGE[s?.state ?? 'unknown'] ?? 'default'}
            text={s ? (STATE_TEXT[s.state] ?? s.state) : '—'}
          />
        </Descriptions.Item>
        <Descriptions.Item label="Источник">{s ? (MODE_TEXT[s.mode] ?? s.mode) : '—'}</Descriptions.Item>
        <Descriptions.Item label="Данные">
          {s ? `${s.split} · с ${s.start ?? 'начала'} · круг ${s.cycle}${s.loop ? ' (по кругу)' : ''}` : '—'}
        </Descriptions.Item>
        <Descriptions.Item label="Скорость">{s ? `×${formatNumber(s.speed)}` : '—'}</Descriptions.Item>
        <Descriptions.Item label="Время данных">{formatClock(s?.data_time)}</Descriptions.Item>
        <Descriptions.Item label="Прогресс дня">
          <Progress percent={Math.round((s?.progress ?? 0) * 100)} size="small" style={{ width: 180 }} />
        </Descriptions.Item>
      </Descriptions>
      <Space wrap size="middle" className="admin-controls">
        <span>Начало дня</span>
        <Select
          value={start}
          onChange={setStart}
          style={{ width: 110 }}
          options={STARTS.map((v) => ({ value: v, label: v }))}
        />
        <span>Скорость</span>
        <Segmented
          value={speed}
          onChange={(v) => setSpeed(Number(v))}
          options={SPEEDS.map((v) => ({ value: v, label: `×${v}` }))}
        />
        <Tooltip title="Через эмулятор — нужен запущенный контейнер эмулятора (make emulator)">
          <Segmented
            value={mode}
            onChange={(v) => setMode(String(v))}
            options={[
              { value: 'ndtp', label: 'NDTP напрямую' },
              { value: 'bridge', label: 'через эмулятор' },
            ]}
          />
        </Tooltip>
        <Popconfirm
          title="Запустить поток заново?"
          description="Часы потока вернутся к началу дня, открытые прогнозы закроются как «сброс»."
          okText="Запустить"
          cancelText="Отмена"
          onConfirm={() =>
            act.mutate({ path: '/api/replay/start', body: { start, speed, mode, loop: true } })
          }
        >
          <Button type="primary" icon={<PlayCircleOutlined />} loading={act.isPending}>
            Запустить заново
          </Button>
        </Popconfirm>
        <Button
          icon={<ReloadOutlined />}
          disabled={!running || speed === s?.speed}
          onClick={() => act.mutate({ path: '/api/replay/speed', body: { speed } })}
        >
          Сменить скорость
        </Button>
        <Popconfirm
          title="Остановить поток?"
          description="Телеметрия перестанет поступать, ТС перейдут в «нет связи»."
          okText="Остановить"
          okButtonProps={{ danger: true }}
          cancelText="Отмена"
          onConfirm={() => act.mutate({ path: '/api/replay/stop' })}
        >
          <Button danger icon={<PauseCircleOutlined />} disabled={!running}>
            Остановить
          </Button>
        </Popconfirm>
      </Space>
      <QueryAlert query={ingest} what="Приём NDTP" />
      <Descriptions
        size="middle"
        column={{ xs: 1, md: 2, xl: 3 }}
        bordered
        title="Приём NDTP (ingest)"
        className="admin-block"
      >
        <Descriptions.Item label="Состояние">
          {i ? (
            <Badge
              status={!i.available ? 'error' : i.listening ? 'success' : 'warning'}
              text={
                !i.available ? 'нет данных от приёмника' : i.listening ? `слушает :${i.port}` : 'не слушает'
              }
            />
          ) : (
            '—'
          )}
        </Descriptions.Item>
        <Descriptions.Item label="Соединения">
          {i ? `${i.connections_active} активных · ${i.connections_total} всего` : '—'}
        </Descriptions.Item>
        <Descriptions.Item label="Пакетов в секунду">
          {i ? formatNumber(i.packets_per_s, 1) : '—'}
        </Descriptions.Item>
        <Descriptions.Item label="Обрывы соединений">
          {i ? formatNumber(i.disconnects) : '—'}
        </Descriptions.Item>
        <Descriptions.Item label="Кадров принято">{i ? formatNumber(i.frames) : '—'}</Descriptions.Item>
        <Descriptions.Item label="Ошибки CRC / разбора">
          {i ? `${formatNumber(i.crc_errors)} / ${formatNumber(i.parse_errors)}` : '—'}
        </Descriptions.Item>
      </Descriptions>
    </div>
  )
}

// ---------------------------------------------------------------- пороги

function ThresholdsTab() {
  const { message } = AntApp.useApp()
  const client = useQueryClient()
  const [form] = Form.useForm<AdminSettings>()
  const settings = useQuery({
    queryKey: ['admin', 'settings'],
    queryFn: ({ signal }) => apiGet<AdminSettings>('/api/admin/settings', undefined, signal),
  })
  useEffect(() => {
    if (settings.data) form.setFieldsValue(settings.data)
  }, [settings.data, form])
  const save = useMutation({
    mutationFn: (body: AdminSettings) => apiPut<AdminSettings>('/api/admin/settings', body),
    onSuccess: (data) => {
      client.setQueryData(['admin', 'settings'], data)
      message.success(`Сохранено: прогнозы применят новые пороги в течение ${data.applies_within_s ?? 30} с`)
    },
    onError: (e) => message.error(errorText(e)),
  })
  if (settings.isError && !settings.data) return <QueryAlert query={settings} what="Пороги" />
  return (
    <div className="admin-tab">
      <p className="admin-tab__hint">
        Цвет ТС на карте: <RiskDot risk="red" /> красный — прогноз опоздания выше красного порога <b>или</b>{' '}
        вероятность опоздания выше красной; <RiskDot risk="green" /> зелёный — ниже обоих зелёных;{' '}
        <RiskDot risk="yellow" /> жёлтый — между ними. Алерт — при открытии инцидента на уровне не ниже
        минимального и его эскалации до красного.
      </p>
      <Form
        form={form}
        layout="vertical"
        className="admin-form"
        onFinish={(values) => save.mutate(values)}
        disabled={settings.isLoading}
      >
        <div className="admin-form__grid">
          <Form.Item
            label="Красный: прогноз опоздания, с"
            name={['risk', 'red_delay_s']}
            rules={[{ required: true }]}
          >
            <InputNumber min={0} max={3600} step={10} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item
            label="Красный: вероятность опоздания"
            name={['risk', 'red_p_late']}
            rules={[{ required: true }]}
          >
            <InputNumber min={0} max={1} step={0.05} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item
            label="Зелёный: прогноз опоздания, с"
            name={['risk', 'green_delay_s']}
            dependencies={[['risk', 'red_delay_s']]}
            rules={[
              { required: true },
              ({ getFieldValue }) => ({
                validator: (_: unknown, value: number | null) => {
                  const red = getFieldValue(['risk', 'red_delay_s']) as number | null
                  return value === null || red === null || value <= red
                    ? Promise.resolve()
                    : Promise.reject(new Error('Не выше красного порога'))
                },
              }),
            ]}
          >
            <InputNumber min={-600} max={3600} step={10} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item
            label="Зелёный: вероятность опоздания"
            name={['risk', 'green_p_late']}
            dependencies={[['risk', 'red_p_late']]}
            rules={[
              { required: true },
              ({ getFieldValue }) => ({
                validator: (_: unknown, value: number | null) => {
                  const red = getFieldValue(['risk', 'red_p_late']) as number | null
                  return value === null || red === null || value <= red
                    ? Promise.resolve()
                    : Promise.reject(new Error('Не выше красной вероятности'))
                },
              }),
            ]}
          >
            <InputNumber min={0} max={1} step={0.05} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item label="Алерт с уровня" name={['alert', 'min_level']}>
            <Select
              options={[
                { value: 'yellow', label: 'жёлтого' },
                { value: 'red', label: 'только красного' },
              ]}
            />
          </Form.Item>
          <Form.Item label="Алерт при вероятности опоздания от" name={['alert', 'min_p_late']}>
            <InputNumber min={0} max={1} step={0.05} style={{ width: '100%' }} />
          </Form.Item>
        </div>
        <Space>
          <Button type="primary" htmlType="submit" loading={save.isPending} icon={<SlidersOutlined />}>
            Сохранить
          </Button>
          <Button onClick={() => settings.data && form.setFieldsValue(settings.data)}>Отменить правки</Button>
          {settings.data?.updated_at ? (
            <span className="admin-tab__muted">
              изменено {new Date(settings.data.updated_at).toLocaleString('ru-RU')}
            </span>
          ) : null}
        </Space>
      </Form>
    </div>
  )
}

// ---------------------------------------------------------------- модели

const FEATURE_TEXT: Record<string, string> = {
  dev_1: 'отклонение на последней подтверждённой остановке',
  cur_dev_s: 'текущее отклонение',
}

function RetrainStatus({ run }: { run: RetrainState | null | undefined }) {
  if (!run || run.state === 'idle') {
    return <span className="admin-tab__muted">переобучение ещё не запускалось</span>
  }
  if (run.state === 'running') {
    return <Badge status="processing" text={`переобучение ${run.base_version ?? ''} идёт…`} />
  }
  if (run.state === 'error') {
    return <Badge status="error" text={`переобучение не удалось: ${run.error ?? 'ошибка'}`} />
  }
  if (!run.version) {
    return (
      <Badge
        status="warning"
        text={
          <>
            поправка не снизила MAE на отложенных {formatNumber(run.n_eval)} прогнозах (
            {formatNumber(run.mae_before_s, 1)} → {formatNumber(run.mae_after_s, 1)} с) — новая версия не
            создана, в работе остаётся <b>{run.base_version}</b>
          </>
        }
      />
    )
  }
  return (
    <Badge
      status="success"
      text={
        <>
          готово: версия <b>{run.version}</b> · MAE на отложенных {formatNumber(run.n_eval)} прогнозах{' '}
          {formatNumber(run.mae_before_s, 1)} → {formatNumber(run.mae_after_s, 1)} с — активируйте её в
          таблице
        </>
      }
    />
  )
}

function FallbackLine({ fallback }: { fallback: FallbackInfo | null | undefined }) {
  if (!fallback) return null
  const share =
    fallback.share === null
      ? 'прогнозов сейчас нет'
      : `сейчас ${formatPercent(fallback.share)} из ${formatNumber(fallback.forecasts)} прогнозов`
  return (
    <p className="admin-tab__hint admin-fallback">
      <b>Fallback</b> — если ml-service недоступен или не ответил за 0,5 с, predictor даёт прогноз сам:{' '}
      {formatNumber(fallback.intercept_s)} с + {formatNumber(fallback.coef, 2)} ×{' '}
      {FEATURE_TEXT[fallback.feature] ?? fallback.feature}; такие прогнозы помечены «fallback», дашборд
      показывает деградацию · {share}
    </p>
  )
}

function ModelsTab() {
  const { message } = AntApp.useApp()
  const client = useQueryClient()
  const models = useQuery({
    queryKey: ['admin', 'models'],
    queryFn: ({ signal }) => apiGet<ModelsOut>('/api/admin/models', undefined, signal),
    // пока идёт переобучение — чаще, чтобы увидеть готовую версию сразу
    refetchInterval: (q) => (q.state.data?.retrain?.state === 'running' ? 2000 : 15_000),
  })
  const run = models.data?.retrain
  const prevState = useRef(run?.state)
  useEffect(() => {
    if (prevState.current === 'running' && run?.state === 'done') {
      if (run.version) message.success(`Переобучение готово: версия ${run.version}`)
      else message.info('Переобучение не улучшило модель — новая версия не создана')
    }
    prevState.current = run?.state
  }, [run, message])
  const activate = useMutation({
    mutationFn: (version: string) =>
      apiPost<ModelsOut>(`/api/admin/models/${encodeURIComponent(version)}/activate`, {}),
    onSuccess: (data) => {
      client.setQueryData(['admin', 'models'], data)
      message.success(`В работе модель ${data.active}`)
    },
    onError: (e) => message.error(errorText(e)),
  })
  const retrain = useMutation({
    mutationFn: () => apiPost<ModelsOut>('/api/admin/models/retrain', {}),
    onSuccess: (data) => {
      client.setQueryData(['admin', 'models'], data)
      prevState.current = 'running'
    },
    onError: (e) => message.error(errorText(e)),
  })
  const mae = (v: number | null) => (v === null || v === undefined ? '—' : `${formatNumber(v, 1)} с`)
  const columns: ColumnsType<ModelVersion> = [
    {
      title: 'Версия',
      dataIndex: 'version',
      render: (v: string, r) => (
        <Space>
          <b>{v}</b>
          {r.active ? <Tag color="success">в работе</Tag> : null}
          {r.online ? (
            <Tooltip
              title={`Поправка версии ${r.online.base} по ${formatNumber(r.online.n_fit)} закрытым прогнозам потока`}
            >
              <Tag>дообучена на потоке</Tag>
            </Tooltip>
          ) : null}
        </Space>
      ),
    },
    {
      title: 'Модель',
      dataIndex: 'model',
      render: (v: string | null, r) => (
        <Tooltip title={r.description}>
          <span>
            {v ?? '—'} {r.precision ? <Tag>{r.precision.toUpperCase()}</Tag> : null}
          </span>
        </Tooltip>
      ),
    },
    { title: 'CV MAE', dataIndex: 'cv_mae', align: 'right', render: mae },
    { title: 'Test MAE', dataIndex: 'test_mae', align: 'right', render: mae },
    { title: 'Baseline cur_dev_s', dataIndex: 'baseline_test_mae', align: 'right', render: mae },
    {
      title: 'Онлайн-MAE на потоке',
      dataIndex: 'online_mae_s',
      align: 'right',
      render: (v: number | null, r) => {
        if (r.active) {
          if (v === null || v === undefined) return 'копится'
          return `${mae(v)}${r.online_closed ? ` · n=${r.online_closed}` : ''}`
        }
        if (r.online) {
          return (
            <Tooltip title={`Проверка на ${formatNumber(r.online.n_eval)} более поздних закрытых прогнозах`}>
              <span>
                {mae(r.online.mae_after_s)}{' '}
                <span className="admin-tab__muted">(было {mae(r.online.mae_before_s)})</span>
              </span>
            </Tooltip>
          )
        }
        return '—'
      },
    },
    {
      title: '',
      key: 'act',
      align: 'right',
      render: (_, r) => (
        <Button
          size="small"
          disabled={r.active}
          loading={activate.isPending && activate.variables === r.version}
          onClick={() => activate.mutate(r.version)}
        >
          Активировать
        </Button>
      ),
    },
  ]
  return (
    <div className="admin-tab">
      <p className="admin-tab__hint">
        Реестр моделей ml-service. <b>holdout</b>-версии обучены только на train — ими честно меряется
        онлайн-MAE на потоке тестового дня; версии без суффикса (train + test) — для сабмита. Переключение —
        без остановки сервиса. <b>Переобучение на потоке</b> подбирает поправку прогноза активной версии по её
        закрытым прогнозам (факт — детектор остановок) и сохраняет её новой версией; поток — тестовый день по
        кругу, поэтому выигрыш на нём оптимистичен.
      </p>
      <QueryAlert query={models} what="Модели" />
      {models.data?.ml === 'down' ? <Alert type="error" showIcon message="ml-service недоступен" /> : null}
      <Space wrap size="middle" className="admin-toolbar">
        <Button
          icon={<SyncOutlined spin={run?.state === 'running'} />}
          loading={retrain.isPending}
          disabled={models.data?.ml !== 'up' || run?.state === 'running'}
          onClick={() => retrain.mutate()}
        >
          Переобучить на потоке
        </Button>
        <RetrainStatus run={run} />
      </Space>
      <Table
        rowKey="version"
        size="middle"
        loading={models.isLoading}
        columns={columns}
        dataSource={models.data?.versions ?? []}
        pagination={false}
      />
      <FallbackLine fallback={models.data?.fallback} />
    </div>
  )
}

// ---------------------------------------------------------------- журнал

type JournalKind = 'alerts' | 'predictions'
type Period = 'all' | '1h' | '3h' | 'custom'

const JOURNAL_LIMIT = 500

const PREDICTION_STATUS: Record<string, string> = {
  open: 'открыт',
  closed: 'закрыт',
  missed: 'без факта',
  expired: 'истёк',
  reset: 'сброс',
}

const SOURCE_TEXT: Record<string, string> = { model: 'модель', fallback: 'fallback' }

/** Границы периода в ISO (время потока) или `undefined` — весь день. */
function periodBounds(
  period: Period,
  streamTime: string | null,
  range: [Dayjs | null, Dayjs | null] | null,
): { from?: string; to?: string } {
  if (period === 'all' || !streamTime) return {}
  const now = Date.parse(streamTime)
  if (period === '1h' || period === '3h') {
    const hours = period === '1h' ? 1 : 3
    return { from: new Date(now - hours * 3_600_000).toISOString() }
  }
  const day = streamTime.slice(0, 10)
  const at = (d: Dayjs | null) => (d ? `${day}T${d.format('HH:mm')}:00Z` : undefined)
  return { from: at(range?.[0] ?? null), to: at(range?.[1] ?? null) }
}

function JournalTab() {
  const [kind, setKind] = useState<JournalKind>('alerts')
  const [trId, setTrId] = useState<number | null>(null)
  const [status, setStatus] = useState<'all' | 'open' | 'closed'>('all')
  const [period, setPeriod] = useState<Period>('all')
  const [range, setRange] = useState<[Dayjs | null, Dayjs | null] | null>(null)
  const streamTime = useLive((s) => s.streamTime)
  const routes = useRoutes()
  const vehicles = useMemo(
    () =>
      (routes.data ?? [])
        .flatMap((r) => r.tr_ids.map((tr) => ({ value: tr, label: `ТС ${tr} · ${r.route_id}` })))
        .sort((a, b) => a.value - b.value),
    [routes.data],
  )
  // период «последний час» считается от часов потока в момент выбора, а не на каждом их тике
  const [periodAt, setPeriodAt] = useState<string | null>(streamTime)
  const bounds = periodBounds(period, period === 'custom' ? streamTime : periodAt, range)
  const params = {
    kind,
    tr_id: trId ?? undefined,
    status: kind === 'predictions' && status !== 'all' ? status : undefined,
    from: bounds.from,
    to: bounds.to,
  }
  const journal = useQuery({
    queryKey: ['admin', 'journal', params],
    queryFn: ({ signal }) =>
      apiGet<JournalOut<AlertOut | PredictionOut>>(
        '/api/admin/journal',
        { ...params, limit: JOURNAL_LIMIT },
        signal,
      ),
    refetchInterval: 10_000,
  })
  const alertColumns: ColumnsType<AlertOut> = [
    { title: 'Выдан', dataIndex: 'issued_at', render: formatClock, width: 100 },
    { title: 'ТС', dataIndex: 'tr_id', width: 90 },
    { title: 'Маршрут', dataIndex: 'route_id', width: 90 },
    {
      title: 'Уровень',
      dataIndex: 'level',
      width: 110,
      render: (v: 'red' | 'yellow') => <RiskTag risk={v} />,
    },
    { title: 'Причина', dataIndex: ['cause', 'text'] },
    { title: 'Остановка', dataIndex: 'target_stop_name' },
    {
      title: 'Прогноз, с',
      dataIndex: 'pred_delay_s',
      align: 'right',
      width: 110,
      render: formatSignedSeconds,
    },
    {
      title: 'Факт, с',
      dataIndex: 'actual_delay_s',
      align: 'right',
      width: 100,
      render: formatSignedSeconds,
    },
  ]
  const predictionColumns: ColumnsType<PredictionOut> = [
    { title: 'План', dataIndex: 'planned_at', render: formatClock, width: 100 },
    { title: 'Выдан', dataIndex: 'issued_at', render: formatClock, width: 100 },
    { title: 'ТС', dataIndex: 'tr_id', width: 90 },
    { title: 'Остановка', dataIndex: 'target_stop_name' },
    {
      title: 'Прогноз, с',
      dataIndex: 'pred_delay_s',
      align: 'right',
      width: 110,
      render: formatSignedSeconds,
    },
    {
      title: 'Факт, с',
      dataIndex: 'actual_delay_s',
      align: 'right',
      width: 100,
      render: formatSignedSeconds,
    },
    {
      title: 'Ошибка, с',
      dataIndex: 'abs_error_s',
      align: 'right',
      width: 110,
      render: (v: number | null) => formatNumber(v),
    },
    {
      title: 'Статус',
      dataIndex: 'status',
      width: 100,
      render: (v: string) => PREDICTION_STATUS[v] ?? v,
    },
    {
      title: 'Источник',
      dataIndex: 'source',
      width: 100,
      render: (v: string | null) => (v ? (SOURCE_TEXT[v] ?? v) : '—'),
    },
  ]
  const csv = buildUrl('/api/admin/journal', { ...params, format: 'csv', limit: 20000 })
  const count = journal.data?.count ?? 0
  const countText =
    count >= JOURNAL_LIMIT
      ? `показаны последние ${formatNumber(count)} (все — в CSV)`
      : `${formatNumber(count)} ${plural(count, 'запись', 'записи', 'записей')}`
  return (
    <div className="admin-tab">
      <Space style={{ marginBottom: 12 }} wrap size="middle">
        <Segmented
          value={kind}
          onChange={(v) => setKind(v as JournalKind)}
          options={[
            { value: 'alerts', label: 'Алерты' },
            { value: 'predictions', label: 'Прогнозы' },
          ]}
        />
        <Select
          allowClear
          showSearch
          placeholder="Все ТС"
          style={{ width: 170 }}
          value={trId}
          onChange={(v: number | undefined) => setTrId(v ?? null)}
          options={vehicles}
          optionFilterProp="label"
        />
        {kind === 'predictions' ? (
          <Segmented
            value={status}
            onChange={(v) => setStatus(v as typeof status)}
            options={[
              { value: 'all', label: 'все' },
              { value: 'open', label: 'открытые' },
              { value: 'closed', label: 'закрытые' },
            ]}
          />
        ) : null}
        <Tooltip
          title={kind === 'alerts' ? 'По времени выдачи алерта' : 'По плановому времени целевой остановки'}
        >
          <Select
            value={period}
            style={{ width: 190 }}
            onChange={(v: Period) => {
              setPeriod(v)
              setPeriodAt(streamTime)
            }}
            options={[
              { value: 'all', label: 'вся линия времени' },
              { value: '1h', label: 'последний час потока', disabled: !streamTime },
              { value: '3h', label: 'последние 3 часа', disabled: !streamTime },
              { value: 'custom', label: 'интервал…', disabled: !streamTime },
            ]}
          />
        </Tooltip>
        {period === 'custom' ? (
          <TimePicker.RangePicker
            format="HH:mm"
            minuteStep={5}
            value={range}
            onChange={(v) => setRange(v as [Dayjs | null, Dayjs | null] | null)}
            placeholder={['с', 'по']}
          />
        ) : null}
        <Button icon={<CloudDownloadOutlined />} href={csv}>
          Скачать CSV
        </Button>
        <span className="admin-tab__muted">текущая линия времени потока · {countText}</span>
      </Space>
      <QueryAlert query={journal} what="Журнал" />
      {kind === 'alerts' ? (
        <Table<AlertOut>
          rowKey="alert_id"
          size="small"
          loading={journal.isLoading}
          columns={alertColumns}
          dataSource={(journal.data?.items ?? []) as AlertOut[]}
          pagination={{ pageSize: 15, showSizeChanger: false }}
        />
      ) : (
        <Table<PredictionOut>
          rowKey="prediction_id"
          size="small"
          loading={journal.isLoading}
          columns={predictionColumns}
          dataSource={(journal.data?.items ?? []) as PredictionOut[]}
          pagination={{ pageSize: 15, showSizeChanger: false }}
        />
      )}
    </div>
  )
}

// ---------------------------------------------------------------- здоровье

const SERVICE_LABEL: Record<string, string> = {
  api: 'API и WebSocket',
  ingest: 'Приём NDTP',
  redis: 'Redis',
  postgres: 'PostgreSQL',
  predictor: 'Predictor',
  'ml-service': 'ML-сервис',
  replayer: 'Демо-поток (replayer)',
}

function HealthTab() {
  const services = useQuery({
    queryKey: ['admin', 'services'],
    queryFn: ({ signal }) => apiGet<ServicesOut>('/api/admin/services', undefined, signal),
    refetchInterval: 5000,
    retry: 0,
  })
  const columns: ColumnsType<ServiceHealth> = [
    { title: 'Сервис', dataIndex: 'name', render: (v: string) => SERVICE_LABEL[v] ?? v },
    {
      title: 'Состояние',
      dataIndex: 'state',
      render: (v: string) => <Badge status={STATE_BADGE[v] ?? 'default'} text={STATE_TEXT[v] ?? v} />,
    },
    { title: 'Подробности', dataIndex: 'detail', render: (v: string | null) => v ?? '—' },
    {
      title: 'Ответ',
      dataIndex: 'latency_ms',
      align: 'right',
      render: (v: number | null) => (v === null ? '—' : `${formatNumber(v, 1)} мс`),
    },
  ]
  const checked = services.data?.checked_at
  return (
    <div className="admin-tab">
      <p className="admin-tab__hint">
        Проверка каждые 5 с{checked ? ` · последняя в ${new Date(checked).toLocaleTimeString('ru-RU')}` : ''}.
        При отказе любого сервиса остальные продолжают работать в режиме деградации.
      </p>
      <QueryAlert query={services} what="Здоровье сервисов (API)" />
      <Table
        rowKey="name"
        size="middle"
        loading={services.isLoading}
        columns={columns}
        dataSource={services.data?.services ?? []}
        pagination={false}
      />
    </div>
  )
}

// ---------------------------------------------------------------- справочники

const LINK_LABEL: Record<string, string> = {
  online: 'на связи',
  stale: 'без свежих данных',
  offline: 'нет связи',
}

function UnitsTable({ routeById }: { routeById: Map<string, RouteOut> }) {
  const units = useQuery({
    queryKey: ['admin', 'units'],
    queryFn: ({ signal }) => apiGet<UnitOut[]>('/api/admin/units', undefined, signal),
    refetchInterval: 10_000,
  })
  // сначала ТС на маршрутах дня (по номеру маршрута), затем остальные
  const rows = useMemo(
    () =>
      [...(units.data ?? [])].sort(
        (a, b) =>
          Number(!a.route_id) - Number(!b.route_id) ||
          (a.route_id ?? '').localeCompare(b.route_id ?? '', 'ru', { numeric: true }) ||
          (a.tr_id ?? 0) - (b.tr_id ?? 0),
      ),
    [units.data],
  )
  const columns: ColumnsType<UnitOut> = [
    {
      title: 'ТС и маршрут',
      key: 'vehicle',
      render: (_: unknown, u) => {
        const route = u.route_id ? routeById.get(u.route_id) : undefined
        return (
          <div className="unit-name">
            <b>{u.tr_id !== null ? `ТС ${u.tr_id}` : `Трекер ${u.unit_id}`}</b>
            <span>
              {route ? (
                <>
                  <span className="inc__route">{route.route_id}</span> {routeEnds(route)}
                </>
              ) : u.route_id ? (
                u.route_id
              ) : u.scheduled === false ? (
                'не в расписании дня'
              ) : u.tr_id === null ? (
                'трекер не сопоставлен с ТС'
              ) : (
                'маршрут не определён'
              )}
            </span>
          </div>
        )
      },
    },
    {
      title: 'Связь',
      dataIndex: 'status',
      render: (v: string) => (
        <Badge
          status={v === 'online' ? 'success' : v === 'stale' ? 'warning' : 'default'}
          text={LINK_LABEL[v] ?? v}
        />
      ),
    },
    { title: 'Риск', dataIndex: 'risk', render: (v: UnitOut['risk']) => (v ? <RiskTag risk={v} /> : '—') },
    {
      title: 'Последние данные',
      dataIndex: 'last_packet_at',
      render: (v: string | null) =>
        v ? (
          <span title={new Date(v).toLocaleString('ru-RU')}>
            {formatIn((Date.parse(v) - Date.now()) / 1000)}
          </span>
        ) : (
          '—'
        ),
    },
    {
      title: 'Трекер (unitId)',
      dataIndex: 'unit_id',
      render: (v: number) => <span className="muted-num">{v}</span>,
    },
  ]
  return (
    <>
      <QueryAlert query={units} what="Справочник ТС" />
      <Table
        rowKey="unit_id"
        size="small"
        loading={units.isLoading}
        columns={columns}
        dataSource={rows}
        pagination={{ pageSize: 20, showSizeChanger: false }}
      />
    </>
  )
}

function RoutesTable({ routes, loading }: { routes: RouteOut[]; loading: boolean }) {
  const columns: ColumnsType<RouteOut> = [
    {
      title: 'Маршрут',
      dataIndex: 'route_id',
      width: 110,
      render: (v: string) => <span className="inc__route">{v}</span>,
    },
    { title: 'Конечные', key: 'ends', render: (_: unknown, r) => routeEnds(r) },
    {
      title: 'Остановок',
      key: 'stops',
      align: 'right',
      width: 120,
      render: (_: unknown, r) => formatNumber(new Set(r.stops.map((s) => s.stop_key)).size),
    },
    {
      title: 'Остановок в рейсе туда и обратно',
      key: 'trip',
      align: 'right',
      width: 200,
      render: (_: unknown, r) => formatNumber(r.stops.length),
    },
    {
      title: 'ТС',
      dataIndex: 'tr_ids',
      render: (v: number[]) => <span className="muted-num">{v.join(', ')}</span>,
    },
  ]
  return (
    <Table
      rowKey="route_id"
      size="small"
      loading={loading}
      columns={columns}
      dataSource={routes}
      pagination={false}
    />
  )
}

interface StopRow {
  stop_key: string
  name: string
  lat: number
  lon: number
  routes: string[]
}

function StopsTable({ routes, loading }: { routes: RouteOut[]; loading: boolean }) {
  const [query, setQuery] = useState('')
  const stops = useMemo(() => {
    const byKey = new Map<string, StopRow>()
    for (const r of routes) {
      for (const s of r.stops) {
        const row = byKey.get(s.stop_key) ?? { ...s, routes: [] }
        if (!row.routes.includes(r.route_id)) row.routes.push(r.route_id)
        byKey.set(s.stop_key, row)
      }
    }
    return [...byKey.values()].sort((a, b) => a.name.localeCompare(b.name, 'ru', { numeric: true }))
  }, [routes])
  const q = query.trim().toLowerCase()
  const rows = q ? stops.filter((s) => s.name.toLowerCase().includes(q)) : stops
  const columns: ColumnsType<StopRow> = [
    { title: 'Остановка', dataIndex: 'name' },
    {
      title: 'Маршруты',
      dataIndex: 'routes',
      render: (v: string[]) => (
        <Space size={4} wrap>
          {v.map((id) => (
            <span key={id} className="inc__route">
              {id}
            </span>
          ))}
        </Space>
      ),
    },
    {
      title: 'Координаты',
      key: 'coords',
      width: 220,
      render: (_: unknown, s) => (
        <span className="muted-num">
          {s.lat.toFixed(5)}, {s.lon.toFixed(5)}
        </span>
      ),
    },
  ]
  return (
    <>
      <Space style={{ marginBottom: 12 }} size="middle">
        <Input.Search
          allowClear
          placeholder="Поиск по названию"
          style={{ width: 280 }}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <span className="admin-tab__muted">
          {formatNumber(rows.length)} {plural(rows.length, 'остановка', 'остановки', 'остановок')}
        </span>
      </Space>
      <Table
        rowKey="stop_key"
        size="small"
        loading={loading}
        columns={columns}
        dataSource={rows}
        pagination={{ pageSize: 15, showSizeChanger: false }}
      />
    </>
  )
}

type Directory = 'units' | 'routes' | 'stops'

function DirectoriesTab() {
  const [view, setView] = useState<Directory>('units')
  const routes = useRoutes()
  const list = useMemo(() => routes.data ?? [], [routes.data])
  const routeById = useMemo(() => new Map(list.map((r) => [r.route_id, r])), [list])
  return (
    <div className="admin-tab">
      <Segmented
        style={{ marginBottom: 16 }}
        value={view}
        onChange={(v) => setView(v as Directory)}
        options={[
          { value: 'units', label: 'ТС (unitId ↔ tr_id)' },
          { value: 'routes', label: `Маршруты · ${list.length}` },
          { value: 'stops', label: 'Остановки' },
        ]}
      />
      {view !== 'units' ? <QueryAlert query={routes} what="Маршруты и остановки" /> : null}
      {view === 'units' ? <UnitsTable routeById={routeById} /> : null}
      {view === 'routes' ? <RoutesTable routes={list} loading={routes.isLoading} /> : null}
      {view === 'stops' ? <StopsTable routes={list} loading={routes.isLoading} /> : null}
    </div>
  )
}

export default function AdminPage() {
  return (
    <div className="page page--admin">
      <div className="page-head">
        <div>
          <div className="page-head__title">Администрирование</div>
          <div className="page-head__sub">
            Поток и приём NDTP, пороги, модели, журнал, здоровье сервисов и справочники · API: <ApiOutlined />{' '}
            /api/admin/* (Swagger /docs)
          </div>
        </div>
      </div>
      <section className="panel admin-panel">
        <Tabs
          size="large"
          destroyOnHidden
          items={[
            {
              key: 'stream',
              label: (
                <span>
                  <PlayCircleOutlined /> Поток
                </span>
              ),
              children: <StreamTab />,
            },
            {
              key: 'thresholds',
              label: (
                <span>
                  <SlidersOutlined /> Пороги
                </span>
              ),
              children: <ThresholdsTab />,
            },
            {
              key: 'models',
              label: (
                <span>
                  <ExperimentOutlined /> Модели
                </span>
              ),
              children: <ModelsTab />,
            },
            {
              key: 'journal',
              label: (
                <span>
                  <FileSearchOutlined /> Журнал
                </span>
              ),
              children: <JournalTab />,
            },
            {
              key: 'health',
              label: (
                <span>
                  <HeartOutlined /> Здоровье
                </span>
              ),
              children: <HealthTab />,
            },
            {
              key: 'directories',
              label: (
                <span>
                  <DatabaseOutlined /> Справочники
                </span>
              ),
              children: <DirectoriesTab />,
            },
          ]}
        />
      </section>
    </div>
  )
}
