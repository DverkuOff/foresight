/** Маршрут по умолчанию для страниц «Нитка» и What-if: выбирается, когда список инцидентов уже получен. */
import { useEffect, useMemo, useState } from 'react'
import type { RouteOut } from '../api/types'
import { sortIncidents } from '../domain/incidents'
import { problemRoute } from '../domain/routes'
import { useLive } from './liveStore'

/** Если backend не отдаёт инциденты, через это время выбираем по тому, что есть. */
const WAIT_MS = 2500

export function useAutoRoute(routes: readonly RouteOut[]): string | null {
  const incidentsReady = useLive((s) => s.incidentsReady)
  const incidentsMap = useLive((s) => s.incidents)
  const [waited, setWaited] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => setWaited(true), WAIT_MS)
    return () => clearTimeout(timer)
  }, [])

  return useMemo(() => {
    if (!routes.length || !(incidentsReady || waited)) return null
    return problemRoute(routes, sortIncidents(incidentsMap.values()))
  }, [routes, incidentsMap, incidentsReady, waited])
}
