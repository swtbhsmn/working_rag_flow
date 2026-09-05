import { useEffect, useMemo, useRef } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { Check, Circle, LoaderCircle, X } from 'lucide-react'
import type { StageEvent } from '../types'

const labels: Record<string, string> = {
  upload: 'Upload file', validation: 'Validate file', extraction: 'Extract text', cleaning: 'Clean text',
  structure: 'Detect structure', segmentation: 'Detect structure', chunking: 'Chunk + overlap',
  tokenization: 'Tokenize chunks', overlap: 'Add overlap', metadata: 'Enrich metadata',
  embedding: 'Create vectors', storage: 'Store vectors', persistence: 'Store vectors', index: 'Build search index',
  query_normalization: 'Normalize query', query_tokenization: 'Tokenize query', query_embedding: 'Embed query',
  comparison: 'Compare vectors', selection: 'Select context', projection: 'Project space', prompt: 'Build prompt',
  prompt_budget: 'Fit context window',
  generation: 'Generate answer', model: 'Inspect model', complete: 'Complete', pipeline: 'Pipeline error',
}

export function PipelineTimeline({ events, active, onSelect }: { events: StageEvent[]; active?: number; onSelect?: (index: number) => void }) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const groups = useMemo(() => {
    const byStage = new Map<string, { stage: string; eventIndexes: number[]; latest: StageEvent }>()
    events.forEach((event, index) => {
      const group = byStage.get(event.stage)
      if (group) {
        group.eventIndexes.push(index)
        group.latest = event
      } else {
        byStage.set(event.stage, { stage: event.stage, eventIndexes: [index], latest: event })
      }
    })
    return [...byStage.values()]
  }, [events])
  const virtualizer = useVirtualizer({
    count: groups.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 58,
    overscan: 8,
  })
  useEffect(() => {
    if (active !== undefined && active >= 0 && active < events.length) {
      const groupIndex = groups.findIndex((group) => group.eventIndexes.includes(active))
      if (groupIndex >= 0) virtualizer.scrollToIndex(groupIndex, { align: 'auto' })
    }
  }, [active, events.length, groups, virtualizer])
  return (
    <div className="timeline-scroll" ref={scrollRef} aria-label="All recorded pipeline events">
      {!events.length && <div className="timeline-empty">Events will appear here as the pipeline runs.</div>}
      <ol className="timeline virtualized" style={{ height: `${virtualizer.getTotalSize()}px` }}>
      {virtualizer.getVirtualItems().map((row) => {
        const group = groups[row.index]
        const event = group.latest
        const stage = group.stage
        const status = event.status
        const firstSequence = events[group.eventIndexes[0]].sequence
        const latestIndex = group.eventIndexes[group.eventIndexes.length - 1]
        const selected = active !== undefined && group.eventIndexes.includes(active)
        const Icon = status === 'failed' ? X : status === 'completed' ? Check : status === 'started' || status === 'progress' ? LoaderCircle : Circle
        return (
          <li key={stage} style={{ height: `${row.size}px`, transform: `translateY(${row.start}px)` }}>
            <button className={selected ? 'active' : ''} onClick={() => onSelect?.(latestIndex)} disabled={!onSelect} aria-label={`${labels[stage] || stage}: ${group.eventIndexes.length} updates, ${status}`}>
              <span className={`timeline-icon ${status}`}><Icon size={14} /></span>
              <span><strong>{labels[stage] || stage}</strong><small>{group.eventIndexes.length > 1 ? `#${firstSequence}–#${event.sequence} · ${group.eventIndexes.length} updates` : `#${event.sequence}`} · {status}</small></span>
            </button>
          </li>
        )
      })}
      </ol>
    </div>
  )
}
