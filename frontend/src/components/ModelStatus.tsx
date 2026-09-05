import { Cpu, Database, RefreshCw } from 'lucide-react'
import type { SystemModels } from '../types'

export function ModelStatus({ models, loading, onRefresh }: { models?: SystemModels; loading: boolean; onRefresh: () => void }) {
  return (
    <section className="model-strip" aria-label="Local model status">
      {(['embedding', 'token_embedding', 'chat'] as const).map((role) => {
        const model = models?.[role]
        const Icon = role === 'chat' ? Cpu : Database
        const label = role === 'token_embedding' ? 'token vectors · none' : `${role} server`
        return (
          <div className="model-chip" key={role}>
            <span className={`status-dot ${model?.ready ? 'ready' : 'offline'}`} />
            <Icon size={16} />
            <div><small>{label}</small><strong>{loading ? 'Checking…' : model?.model_id?.split('/').pop() || 'Unavailable'}</strong></div>
            {model?.embedding_dimensions && <code>{model.embedding_dimensions}d</code>}
          </div>
        )
      })}
      <button className="icon-button" onClick={onRefresh} aria-label="Refresh model status"><RefreshCw size={16} /></button>
    </section>
  )
}
