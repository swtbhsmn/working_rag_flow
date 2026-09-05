import { useEffect, useRef, useState } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { ArrowDown, ArrowRight, Braces, CheckCircle2, CircleDot, Database, LoaderCircle, Network, Sigma } from 'lucide-react'
import { api } from '../api'
import type { ContextualToken } from '../types'

function VectorViewer({ vector, label }: { vector: number[]; label: string }) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const columns = 4
  const rows = useVirtualizer({ count: Math.ceil(vector.length / columns), getScrollElement: () => scrollRef.current, estimateSize: () => 37, overscan: 5 })
  return <div className="dimension-viewer" ref={scrollRef} tabIndex={0} aria-label={`${label}, ${vector.length} dimensions`}>
    <div style={{ height: `${rows.getTotalSize()}px`, position: 'relative' }}>
      {rows.getVirtualItems().map((row) => <div className="dimension-row" key={row.key} style={{ transform: `translateY(${row.start}px)` }}>
        {vector.slice(row.index * columns, row.index * columns + columns).map((value, offset) => {
          const index = row.index * columns + offset
          return <code key={index}><i>d{index}</i><span>{Number(value).toFixed(6)}</span></code>
        })}
      </div>)}
    </div>
  </div>
}

export function TokenEmbeddingGraph({
  tokenIds, input, vector, dimensions, rawClsVector = [], rawClsNorm = 0, returnedNorm = 0, finalNorm = 0,
}: {
  tokenIds: number[]
  input: string
  vector: number[]
  dimensions?: number
  rawClsVector?: number[]
  rawClsNorm?: number
  returnedNorm?: number
  finalNorm?: number
}) {
  const tokenScrollRef = useRef<HTMLDivElement>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [vectors, setVectors] = useState<Record<number, ContextualToken>>({})
  const [loading, setLoading] = useState<number | null>(null)
  const [error, setError] = useState('')
  const tokens = useVirtualizer({ horizontal: true, count: tokenIds.length, getScrollElement: () => tokenScrollRef.current, estimateSize: () => 88, overscan: 8 })

  useEffect(() => { setSelected(null); setVectors({}); setError('') }, [input])

  async function inspectToken(index: number) {
    setSelected(index); setError('')
    if (vectors[index]) return
    setLoading(index)
    try {
      const result = await api.contextualToken(input, index, tokenIds)
      setVectors((current) => ({ ...current, [index]: result }))
    } catch (reason) { setError((reason as Error).message) }
    finally { setLoading(null) }
  }

  if (!tokenIds.length) return null
  const contextual = selected === null ? undefined : vectors[selected]
  const expectedDimensions = dimensions || vector.length

  return <section className="token-embedding-graph">
    <header>
      <div><span className="eyebrow">Real embedding pipeline</span><h3>How token IDs become one embedding</h3></div>
      <span className="graph-count">{tokenIds.length} tokens → 1 × {expectedDimensions}</span>
    </header>

    <div className="embedding-flow" aria-label="Embedding pipeline stages">
      <div className="flow-stage"><small>01</small><strong>Token IDs</strong><span>N integer IDs</span></div><ArrowRight />
      <div className="flow-stage"><small>02</small><strong>Embedding lookup</strong><span>IDs → initial vectors</span></div><ArrowRight />
      <div className="flow-stage live-none"><small>03 · :8082</small><strong>Contextualization</strong><span>N × {expectedDimensions} · pooling none</span></div><ArrowRight />
      <div className="flow-stage"><small>04 · main server</small><strong>CLS pooling</strong><span>N × {expectedDimensions} → 1 × {expectedDimensions}</span></div><ArrowRight />
      <div className="flow-stage"><small>05</small><strong>L2 normalization</strong><span>v / ||v||₂</span></div><ArrowRight />
      <div className="flow-stage final"><small>06</small><strong>Final vector</strong><span>1 × {expectedDimensions}</span></div><ArrowRight />
      <div className="flow-stage database"><small>07</small><strong>Vector DB</strong><span>cosine similarity search</span></div>
    </div>

    <div className="pipeline-detail">
      <div className="detail-heading"><div><CircleDot size={16} /><span><strong>Token IDs</strong><small>Click a token to load its real contextual vector.</small></span></div><code>shape [{tokenIds.length}]</code></div>
      <div className="token-lane" ref={tokenScrollRef} tabIndex={0} aria-label={`Scrollable list of ${tokenIds.length} clickable token IDs`}>
        <div className="token-canvas" style={{ width: `${tokens.getTotalSize()}px` }}>
          {tokens.getVirtualItems().map((item) => <button className={`token-node ${selected === item.index ? 'selected' : ''}`} key={item.key}
            style={{ width: `${item.size - 8}px`, transform: `translateX(${item.start + 4}px)` }} onClick={() => inspectToken(item.index)}
            aria-label={`Inspect token ${item.index}, ID ${tokenIds[item.index]}`}>
            <small>t{item.index}</small><strong>{tokenIds[item.index]}</strong>{vectors[item.index] && <CheckCircle2 size={11} />}
          </button>)}
        </div>
      </div>
    </div>

    <ArrowDown className="vertical-flow-arrow" />
    <div className="pipeline-detail contextual-panel">
      <div className="detail-heading"><div><Network size={16} /><span><strong>Contextual token vectors</strong><small>Real, unnormalized output from the pooling-none server on port 8082.</small></span></div><code>shape [{tokenIds.length}, {expectedDimensions}]</code></div>
      {selected === null && <div className="vector-placeholder">Select a token above. No contextual values are generated or guessed by this UI.</div>}
      {loading !== null && <div className="vector-placeholder"><LoaderCircle className="spin" size={17} />Reading all contextual vectors, then selecting t{loading}…</div>}
      {error && <div className="contextual-error" role="alert">{error}</div>}
      {contextual && <div className="selected-vector">
        <div className="selected-vector-meta"><span><strong>t{contextual.token_index}</strong><small>index</small></span><span><strong>{contextual.token_id}</strong><small>token ID</small></span><span><strong>{JSON.stringify(contextual.token_piece)}</strong><small>token piece</small></span><span><strong>{contextual.dimensions}</strong><small>dimensions</small></span></div>
        <VectorViewer vector={contextual.vector} label={`Contextual vector for token ${contextual.token_index}`} />
        <p>Source: <code>{contextual.server_url}</code> · pooling <code>none</code> · unnormalized. These values are not the vector stored or searched.</p>
      </div>}
    </div>

    <ArrowDown className="vertical-flow-arrow" />
    <div className="pool-normalize-grid">
      <div className="operation-card"><Sigma size={18} /><div><small>CLS POOLING · MAIN SERVER</small><strong>{tokenIds.length} × {expectedDimensions} → 1 × {expectedDimensions}</strong><p>The model selects its CLS pooled representation. This is a separate operation from normalization.</p></div></div>
      <div className="operation-card"><Braces size={18} /><div><small>L2 NORMALIZATION</small><strong>v / ||v||₂</strong><p>{rawClsVector.length ? `Raw CLS norm: ${rawClsNorm.toFixed(6)}.` : 'The raw CLS vector was unavailable; no values are inferred.'} Search endpoint return norm: {returnedNorm.toFixed(6)}. Final norm: {finalNorm.toFixed(6)}.</p></div></div>
    </div>

    <ArrowDown className="vertical-flow-arrow" />
    <div className="final-query-vector">
      <div className="final-vector-title"><span className="embedding-icon"><Network size={20} /></span><div><small>SEARCH VECTOR</small><strong>Final query embedding — {expectedDimensions} dimensions</strong><p>This exact L2-normalized vector is used for cosine similarity search.</p></div><Database size={22} /></div>
      <VectorViewer vector={vector} label="Final query embedding used for similarity search" />
    </div>
  </section>
}
