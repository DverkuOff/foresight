/** REST-запросы (TanStack Query) по контракту `docs/api-contract.md`. */
import { useMutation, useQuery } from '@tanstack/react-query'
import { apiGet, apiPost } from './client'
import type {
  AlertListOut,
  HealthOut,
  HorizonOut,
  IncidentDetailOut,
  IncidentListOut,
  PerfOut,
  PredictionListOut,
  RouteOut,
  StringlineOut,
  VehicleListOut,
  WhatIfOut,
  WhatIfRequest,
} from './types'
import { normalizeIncident } from './wsMessages'

export const queryKeys = {
  routes: ['routes'] as const,
  vehicles: ['vehicles'] as const,
  incidents: (status: 'active' | 'all') => ['incidents', status] as const,
  incident: (id: string) => ['incident', id] as const,
  alerts: ['alerts'] as const,
  predictionsClosed: ['predictions', 'closed'] as const,
  stringline: (routeId: string, from: string, to: string) => ['stringline', routeId, from, to] as const,
  horizon: (windowS: number) => ['horizon', windowS] as const,
  perf: ['perf'] as const,
  health: ['health'] as const,
}

export const fetchRoutes = (signal?: AbortSignal) => apiGet<RouteOut[]>('/api/routes', undefined, signal)
export const fetchVehicles = (signal?: AbortSignal) =>
  apiGet<VehicleListOut>('/api/vehicles', undefined, signal)
export async function fetchIncidents(status: 'active' | 'all' = 'active', signal?: AbortSignal) {
  const res = await apiGet<IncidentListOut>('/api/incidents', { status, limit: 200 }, signal)
  return { ...res, items: res.items.map(normalizeIncident) }
}
export const fetchAlerts = (signal?: AbortSignal) =>
  apiGet<AlertListOut>('/api/alerts', { limit: 300 }, signal)
export const fetchClosedPredictions = (signal?: AbortSignal) =>
  apiGet<PredictionListOut>('/api/predictions', { status: 'closed', limit: 100 }, signal)

export function useRoutes() {
  return useQuery({
    queryKey: queryKeys.routes,
    queryFn: ({ signal }) => fetchRoutes(signal),
    staleTime: 5 * 60_000,
    refetchInterval: 5 * 60_000,
  })
}

export function useIncident(id: string | null) {
  return useQuery({
    queryKey: queryKeys.incident(id ?? ''),
    queryFn: async ({ signal }): Promise<IncidentDetailOut> => {
      const detail = await apiGet<IncidentDetailOut>(
        `/api/incidents/${encodeURIComponent(id ?? '')}`,
        undefined,
        signal,
      )
      return { ...detail, incident_id: String(detail.incident_id) }
    },
    enabled: id !== null,
    refetchInterval: 5_000,
    retry: 1,
  })
}

export function useStringline(routeId: string | null, from: string | null, to: string | null) {
  return useQuery({
    queryKey: queryKeys.stringline(routeId ?? '', from ?? '', to ?? ''),
    queryFn: ({ signal }) =>
      apiGet<StringlineOut>('/api/stringline', { route_id: routeId, from, to }, signal),
    enabled: routeId !== null && from !== null && to !== null,
    refetchInterval: 10_000,
    placeholderData: (prev) => prev,
  })
}

export function useHorizon(windowS = 3600) {
  return useQuery({
    queryKey: queryKeys.horizon(windowS),
    queryFn: ({ signal }) => apiGet<HorizonOut>('/api/metrics/horizon', { window_s: windowS }, signal),
    refetchInterval: 5_000,
    placeholderData: (prev) => prev,
  })
}

export function usePerf() {
  return useQuery({
    queryKey: queryKeys.perf,
    queryFn: ({ signal }) => apiGet<PerfOut>('/api/metrics/perf', undefined, signal),
    refetchInterval: 3_000,
    placeholderData: (prev) => prev,
  })
}

export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: ({ signal }) => apiGet<HealthOut>('/health', undefined, signal),
    refetchInterval: 10_000,
    retry: 0,
  })
}

export function useWhatIf() {
  return useMutation({
    mutationFn: (request: WhatIfRequest) => apiPost<WhatIfOut>('/api/whatif', request),
  })
}
