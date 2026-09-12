import type {
  DatasetMeta,
  MoleculeDetail,
  MoleculeSummary,
  Review,
  ReviewStats,
  ReviewUpdate,
} from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    throw new Error(payload?.detail ?? `Request failed: ${response.status}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  meta: () => request<DatasetMeta>('/api/meta'),
  stats: () => request<ReviewStats>('/api/stats'),
  items: (params: URLSearchParams) =>
    request<{ items: MoleculeSummary[]; total: number }>(`/api/items?${params}`),
  item: (id: string) => request<MoleculeDetail>(`/api/items/${encodeURIComponent(id)}`),
  saveReview: (id: string, review: ReviewUpdate) =>
    request<Review>(`/api/items/${encodeURIComponent(id)}/review`, {
      method: 'PUT',
      body: JSON.stringify(review),
    }),
}
