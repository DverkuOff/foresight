import {
  ApiOutlined,
  AreaChartOutlined,
  DashboardOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  ReadOutlined,
  SafetyCertificateOutlined,
  SettingOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import { Layout, Menu } from 'antd'
import { Suspense, useEffect, useMemo } from 'react'
import { Outlet, useLocation, useNavigate } from 'react-router'
import type { SocketFactory } from '../api/ws'
import { config } from '../config'
import { useLiveFeed } from '../state/useLiveFeed'
import { HeaderStatus } from './HeaderStatus'
import { PageSpinner } from './PageSpinner'

const { Sider, Header, Content } = Layout

const NAV = [
  { key: '/', icon: <DashboardOutlined />, label: 'Оперативная обстановка' },
  { key: '/stringline', icon: <LineChartOutlined />, label: 'График движения' },
  { key: '/honesty', icon: <SafetyCertificateOutlined />, label: 'Честность прогноза' },
  { key: '/whatif', icon: <ExperimentOutlined />, label: 'What-if' },
  { key: '/perf', icon: <ThunderboltOutlined />, label: 'Производительность' },
  { key: '/admin', icon: <SettingOutlined />, label: 'Администрирование' },
]

/** Внешние ссылки внизу меню: открываются в новой вкладке. */
const LINKS = [
  { key: 'swagger', icon: <ApiOutlined />, label: 'Swagger API', href: '/docs' },
  { key: 'grafana', icon: <AreaChartOutlined />, label: 'Grafana', href: `${config.grafanaUrl}/` },
  {
    key: 'docs',
    icon: <ReadOutlined />,
    label: 'Документация',
    href: 'https://dverkuoff.github.io/foresight/',
  },
]

function Logo() {
  return (
    <div className="app-logo">
      <svg className="app-logo__mark" viewBox="0 0 64 64" aria-hidden="true">
        <circle className="app-logo__ring" cx="32" cy="32" r="17" fill="none" strokeWidth="5" />
        <circle cx="32" cy="32" r="6" fill="#22c55e" />
        <path
          className="app-logo__ring"
          d="M32 6v8M32 50v8M6 32h8M50 32h8"
          strokeWidth="4"
          strokeLinecap="round"
        />
      </svg>
      <span className="app-logo__name">Форсайт</span>
    </div>
  )
}

export function AppLayout({ createSocket }: { createSocket: SocketFactory }) {
  useLiveFeed(createSocket)
  const location = useLocation()
  const navigate = useNavigate()
  const selected = useMemo(() => {
    const match = NAV.filter((n) => n.key !== '/' && location.pathname.startsWith(n.key))
    return match.length ? match[0].key : '/'
  }, [location.pathname])
  const pageTitle = NAV.find((n) => n.key === selected)?.label ?? ''
  // два продукта в одном фронтенде (docs/architecture.md §0): экран диспетчера и админка
  const admin = selected === '/admin'
  const product = admin ? 'Форсайт · Администрирование' : 'Форсайт · Диспетчер'
  useEffect(() => {
    document.title = product
  }, [product])

  return (
    <Layout className="app-shell">
      <Sider breakpoint="xxl" collapsedWidth={64} width={244} theme="light" trigger={null}>
        <Logo />
        <Menu
          theme="light"
          mode="inline"
          selectedKeys={[selected]}
          items={NAV.map((n) => ({ key: n.key, icon: n.icon, label: n.label, title: n.label }))}
          onClick={({ key }) => navigate(key)}
          style={{ borderInlineEnd: 'none', paddingTop: 8 }}
        />
        <Menu
          theme="light"
          mode="inline"
          selectable={false}
          className="app-links"
          aria-label="Ссылки"
          items={LINKS.map((l) => ({
            key: l.key,
            icon: l.icon,
            title: l.label,
            label: (
              <a href={l.href} target="_blank" rel="noreferrer">
                {l.label}
              </a>
            ),
          }))}
          style={{ borderInlineEnd: 'none' }}
        />
      </Sider>
      <Layout>
        <Header className="app-header">
          <div className="app-header__title">
            <span className="app-header__product">{product}</span>
            <span className="app-header__page">{admin ? 'управление системой' : pageTitle}</span>
          </div>
          <div className="app-header__spacer" />
          <HeaderStatus />
        </Header>
        <Content>
          <Suspense fallback={<PageSpinner />}>
            <Outlet />
          </Suspense>
        </Content>
      </Layout>
    </Layout>
  )
}
