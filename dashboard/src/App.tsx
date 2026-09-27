import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { App as AntApp, ConfigProvider } from 'antd'
import ruRU from 'antd/locale/ru_RU'
import { lazy, useState } from 'react'
import { createBrowserRouter, RouterProvider } from 'react-router'
import type { SocketFactory } from './api/ws'
import { AppLayout } from './components/AppLayout'
import { antdTheme } from './theme'

const OverviewPage = lazy(() => import('./pages/OverviewPage'))
const StringlinePage = lazy(() => import('./pages/StringlinePage'))
const HonestyPage = lazy(() => import('./pages/HonestyPage'))
const WhatIfPage = lazy(() => import('./pages/WhatIfPage'))
const PerfPage = lazy(() => import('./pages/PerfPage'))
const AdminPage = lazy(() => import('./pages/AdminPage'))
const NotFoundPage = lazy(() => import('./pages/NotFoundPage'))

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { refetchOnWindowFocus: false, retry: 1, staleTime: 2_000 },
  },
})

export function App({ createSocket }: { createSocket: SocketFactory }) {
  const [router] = useState(() =>
    createBrowserRouter([
      {
        path: '/',
        element: <AppLayout createSocket={createSocket} />,
        children: [
          { index: true, element: <OverviewPage /> },
          { path: 'stringline', element: <StringlinePage /> },
          { path: 'honesty', element: <HonestyPage /> },
          { path: 'whatif', element: <WhatIfPage /> },
          { path: 'perf', element: <PerfPage /> },
          { path: 'admin/*', element: <AdminPage /> },
          { path: '*', element: <NotFoundPage /> },
        ],
      },
    ]),
  )
  return (
    <ConfigProvider theme={antdTheme} locale={ruRU}>
      <AntApp>
        <QueryClientProvider client={queryClient}>
          <RouterProvider router={router} />
        </QueryClientProvider>
      </AntApp>
    </ConfigProvider>
  )
}
