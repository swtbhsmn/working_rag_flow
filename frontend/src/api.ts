import type { ContextualToken, DocumentSummary, StageEvent, SystemModels } from './types'

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options)
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(body.detail || 'Request failed')
  }
  return response.status === 204 ? (undefined as T) : response.json()
}

export const api = {
  models: () => request<SystemModels>('/api/system/models'),
  contextualToken: (input: string, tokenIndex: number, expectedTokenIds: number[]) => request<ContextualToken>('/api/embeddings/contextual-token', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input, token_index: tokenIndex, expected_token_ids: expectedTokenIds }),
  }),
  documents: () => request<DocumentSummary[]>('/api/documents'),
  upload: (file: File, chunkSize: number, overlap: number) => {
    const body = new FormData()
    body.set('file', file)
    body.set('chunk_size', String(chunkSize))
    body.set('chunk_overlap', String(overlap))
    return request<{ document_id: string; job_id: string }>('/api/documents', { method: 'POST', body })
  },
  replay: (id: string) => request<{ events: StageEvent[] }>(`/api/documents/${id}/pipeline`),
  reindex: (id: string, chunkSize: number, overlap: number) => request<{ document_id: string; job_id: string }>(`/api/documents/${id}/reindex?chunk_size=${chunkSize}&chunk_overlap=${overlap}`, { method: 'POST' }),
  remove: (id: string) => request<void>(`/api/documents/${id}`, { method: 'DELETE' }),
  search: (body: object) => request<{ search_id: string }>('/api/searches', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }),
  transformer: (body: object) => request<{ run_id: string }>('/api/transformer/runs', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }),
}

export function subscribe(url: string, onEvent: (event: StageEvent) => void, onEnd: () => void, onError: (message: string) => void) {
  const source = new EventSource(url)
  source.addEventListener('stage', (message) => onEvent(JSON.parse((message as MessageEvent).data)))
  source.addEventListener('end', () => { source.close(); onEnd() })
  source.addEventListener('error', (event) => {
    if (source.readyState === EventSource.CLOSED) return
    source.close()
    onError(event instanceof MessageEvent ? event.data : 'The event stream was interrupted')
  })
  return () => source.close()
}
