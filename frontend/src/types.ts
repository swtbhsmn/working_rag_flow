export type StageStatus = 'started' | 'progress' | 'completed' | 'failed'

export type StageEvent = {
  stage: string
  status: StageStatus
  sequence: number
  timestamp: string
  payload: Record<string, unknown>
}

export type ModelInfo = {
  role: 'chat' | 'embedding' | 'token_embedding'
  ready: boolean
  base_url: string
  model_id?: string
  context_size?: number
  embedding_dimensions?: number
  error?: string
}

export type SystemModels = { chat: ModelInfo; embedding: ModelInfo; token_embedding: ModelInfo }

export type ContextualToken = {
  token_index: number
  token_id: number
  token_piece: string
  token_count: number
  dimensions: number
  vector: number[]
  server_url: string
  pooling: 'none'
  normalized: false
}

export type DocumentSummary = {
  id: string
  filename: string
  media_type: string
  size_bytes: number
  status: string
  chunk_count: number
  token_count: number
  chunk_size: number
  chunk_overlap: number
  model_id?: string
  embedding_dim?: number
  reindex_required: boolean
  created_at: string
  error?: string
}

export type RankedChunk = {
  rank: number
  chunk_id: string
  document_id: string
  filename: string
  page: number
  chunk_index: number
  text: string
  token_count: number
  score: number
  marker: string
}

export type ProjectionPoint = { id: string; x: number; y: number; chunk_id?: string; filename?: string; score?: number }
