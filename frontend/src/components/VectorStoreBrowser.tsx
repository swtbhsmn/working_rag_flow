import { useRef } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'

export type StoredVectorRecord = {
  id: string
  document_id: string
  chunk_index: number
  page: number
  text: string
  token_count: number
  embedding_dimensions: number
  vector: number[]
}

export function VectorDimensions({ vector }: { vector: number[] }) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const virtualizer = useVirtualizer({
    horizontal: true,
    count: vector.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 82,
    overscan: 6,
  })
  return (
    <div className="vector-dimensions" ref={scrollRef} aria-label={`Vector dimensions d0 through d${Math.max(0, vector.length - 1)}`}>
      <div style={{ width: `${virtualizer.getTotalSize()}px` }}>
        {virtualizer.getVirtualItems().map((item) => <code key={item.index} style={{ width: `${item.size - 5}px`, transform: `translateX(${item.start}px)` }}><i>d{item.index}</i>{vector[item.index].toFixed(8)}</code>)}
      </div>
    </div>
  )
}

export function VectorStoreBrowser({ records }: { records: StoredVectorRecord[] }) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const virtualizer = useVirtualizer({
    count: records.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 186,
    overscan: 4,
  })

  return (
    <div className="vector-store-browser" ref={scrollRef} aria-label={`All ${records.length} stored vector records`}>
      <div className="vector-store-canvas" style={{ height: `${virtualizer.getTotalSize()}px` }}>
        {virtualizer.getVirtualItems().map((row) => {
          const record = records[row.index]
          return (
            <article
              className="stored-vector-row"
              key={record.id}
              ref={virtualizer.measureElement}
              data-index={row.index}
              style={{ transform: `translateY(${row.start}px)` }}
            >
              <header><strong>Chunk {record.chunk_index + 1}</strong><span>page {record.page} · {record.token_count} tokens</span><code>{record.id}</code></header>
              <p>{record.text}</p>
              <div className="vector-heading"><small>Full vector · {record.embedding_dimensions} dimensions</small><span>scroll d0 → d{record.embedding_dimensions - 1}</span></div>
              <VectorDimensions vector={record.vector} />
            </article>
          )
        })}
      </div>
    </div>
  )
}
