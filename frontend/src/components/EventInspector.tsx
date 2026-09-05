import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import type { ProjectionPoint, StageEvent } from '../types'
import { VectorPlot } from './VectorPlot'

function renderValue(value: unknown): string {
  if (typeof value === 'string') return value
  return JSON.stringify(value, null, 2)
}

function isProjection(value: unknown): value is ProjectionPoint[] {
  return Array.isArray(value) && value.length > 0 && value.every((point) => (
    typeof point === 'object' && point !== null
    && typeof (point as ProjectionPoint).id === 'string'
    && typeof (point as ProjectionPoint).x === 'number'
    && typeof (point as ProjectionPoint).y === 'number'
  ))
}

function usesFullWidth(key: string, value: unknown): boolean {
  if (['answer', 'output', 'prompt', 'message_content', 'token_ids', 'vector', 'raw_cls_vector', 'candidates', 'selected'].includes(key)) return true
  return typeof value !== 'number' && typeof value !== 'boolean' && renderValue(value).length > 240
}

export function EventInspector({ event, events = [], activeIndex, onSelect }: { event?: StageEvent; events?: StageEvent[]; activeIndex?: number; onSelect?: (index: number) => void }) {
  if (!event) return <div className="empty-inspector">Run or replay a pipeline to inspect its real inputs and outputs.</div>
  const relatedIndexes = events.map((item, index) => item.stage === event.stage ? index : -1).filter((index) => index >= 0)
  const updatePosition = activeIndex === undefined ? -1 : relatedIndexes.indexOf(activeIndex)
  const payloadEntries = Object.entries(event.payload).sort(([leftKey, leftValue], [rightKey, rightValue]) => Number(usesFullWidth(leftKey, leftValue)) - Number(usesFullWidth(rightKey, rightValue)))
  return (
    <article className="inspector">
      <header><span className="eyebrow">Stage {event.sequence}</span><h3>{event.stage.replaceAll('_', ' ')}</h3><span className={`badge ${event.status}`}>{event.status}</span></header>
      {relatedIndexes.length > 1 && updatePosition >= 0 && <div className="update-navigator">
        <button disabled={updatePosition === 0} onClick={() => onSelect?.(relatedIndexes[updatePosition - 1])} aria-label="Previous stage update"><ChevronLeft size={14} /></button>
        <label><span>Update {updatePosition + 1} of {relatedIndexes.length}</span><input type="range" min="0" max={relatedIndexes.length - 1} value={updatePosition} onChange={(change) => onSelect?.(relatedIndexes[Number(change.target.value)])} /></label>
        <button disabled={updatePosition === relatedIndexes.length - 1} onClick={() => onSelect?.(relatedIndexes[updatePosition + 1])} aria-label="Next stage update"><ChevronRight size={14} /></button>
      </div>}
      <div className="payload-grid">
        {payloadEntries.map(([key, value]) => (
          <div className={`payload ${usesFullWidth(key, value) ? 'payload-wide' : ''}`} key={key}>
            <small>{key.replaceAll('_', ' ')}</small>
            {isProjection(value)
              ? <div className="projection-payload"><VectorPlot points={value} /><details><summary>View coordinate JSON</summary><pre>{renderValue(value)}</pre></details></div>
              : typeof value === 'string' && (key === 'output' || key === 'answer')
              ? <div className="markdown-answer payload-markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{value}</ReactMarkdown></div>
              : <pre>{renderValue(value)}</pre>}
          </div>
        ))}
      </div>
    </article>
  )
}
