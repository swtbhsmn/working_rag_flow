import { useEffect, useMemo, useRef, useState } from 'react'
import { BookOpen, FileText, PauseCircle, Play, PlayCircle, RotateCcw, Search, SlidersHorizontal, StepForward, Trash2, UploadCloud } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api, subscribe } from '../api'
import type { DocumentSummary, ProjectionPoint, RankedChunk, StageEvent } from '../types'
import { EventInspector } from './EventInspector'
import { PipelineTimeline } from './PipelineTimeline'
import { TokenEmbeddingGraph } from './TokenEmbeddingGraph'
import { VectorPlot } from './VectorPlot'
import { VectorStoreBrowser, type StoredVectorRecord } from './VectorStoreBrowser'

export function RagExplorer() {
  const [workflow, setWorkflow] = useState<'knowledge' | 'questions'>('knowledge')
  const [documents, setDocuments] = useState<DocumentSummary[]>([])
  const [selected, setSelected] = useState<string[]>([])
  const [events, setEvents] = useState<StageEvent[]>([])
  const [activeEvent, setActiveEvent] = useState(0)
  const [chunkSize, setChunkSize] = useState(256)
  const [overlap, setOverlap] = useState(32)
  const [query, setQuery] = useState('What are the main ideas in these documents?')
  const [topK, setTopK] = useState(5)
  const [threshold, setThreshold] = useState(0.2)
  const [generate, setGenerate] = useState(true)
  const [busy, setBusy] = useState(false)
  const [autoFollow, setAutoFollow] = useState(true)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [stepMode, setStepMode] = useState(true)
  const [playbackPaused, setPlaybackPaused] = useState(false)
  const [bufferedCount, setBufferedCount] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const autoFollowRef = useRef(true)
  const stepModeRef = useRef(true)
  const playbackPausedRef = useRef(false)
  const bufferedEventsRef = useRef<StageEvent[]>([])

  const changeAutoFollow = (value: boolean) => {
    autoFollowRef.current = value
    setAutoFollow(value)
  }

  const inspectEvent = (index: number) => {
    changeAutoFollow(false)
    setActiveEvent(index)
  }

  const refresh = async () => {
    const docs = await api.documents()
    setDocuments(docs)
    setSelected((current) => current.filter((id) => docs.some((doc) => doc.id === id && doc.status === 'ready')))
  }
  useEffect(() => { refresh().catch((err) => setError(err.message)) }, [])

  const showEvents = (incoming: StageEvent[]) => {
    if (!incoming.length) return
    setEvents((current) => {
      const next = [...current, ...incoming]
      if (autoFollowRef.current) setActiveEvent(next.length - 1)
      return next
    })
  }

  const setPhasePaused = (value: boolean) => {
    playbackPausedRef.current = value
    setPlaybackPaused(value)
  }

  const resetPlayback = () => {
    bufferedEventsRef.current = []
    setBufferedCount(0)
    setPhasePaused(false)
  }

  const pushEvent = (event: StageEvent) => {
    if (event.status === 'failed') {
      setBusy(false)
      setError(String(event.payload.message || `${event.stage.replaceAll('_', ' ')} failed`))
    }
    if (stepModeRef.current && playbackPausedRef.current) {
      bufferedEventsRef.current.push(event)
      setBufferedCount(bufferedEventsRef.current.length)
      return
    }
    showEvents([event])
    if (stepModeRef.current && (event.status === 'completed' || event.status === 'failed')) setPhasePaused(true)
  }

  const revealNextPhase = () => {
    setPhasePaused(false)
    const next: StageEvent[] = []
    while (bufferedEventsRef.current.length) {
      const event = bufferedEventsRef.current.shift()!
      next.push(event)
      if (event.status === 'completed' || event.status === 'failed') break
    }
    setBufferedCount(bufferedEventsRef.current.length)
    showEvents(next)
    if (next.some((event) => event.status === 'completed' || event.status === 'failed')) setPhasePaused(true)
  }

  const resumeLive = () => {
    stepModeRef.current = false
    setStepMode(false)
    setPhasePaused(false)
    const pending = bufferedEventsRef.current.splice(0)
    setBufferedCount(0)
    showEvents(pending)
  }

  const enableStepMode = () => {
    stepModeRef.current = true
    setStepMode(true)
    setPhasePaused(events.length > 0)
  }

  async function upload(file?: File) {
    if (!file) return
    setError(''); setEvents([]); setBusy(true); changeAutoFollow(true); resetPlayback()
    try {
      const result = await api.upload(file, chunkSize, overlap)
      subscribe(`/api/jobs/${result.job_id}/events`, pushEvent, () => { setBusy(false); refresh() }, (message) => { setBusy(false); setError(message) })
      await refresh()
    } catch (err) { setBusy(false); setError((err as Error).message) }
  }

  async function runSearch() {
    if (!selected.length) return setError('Select at least one ready document.')
    setError(''); setEvents([]); setBusy(true); changeAutoFollow(true); resetPlayback()
    try {
      const result = await api.search({ query, document_ids: selected, top_k: topK, similarity_threshold: threshold, generate_answer: generate })
      subscribe(`/api/searches/${result.search_id}/events`, pushEvent, () => setBusy(false), (message) => { setBusy(false); setError(message) })
    } catch (err) { setBusy(false); setError((err as Error).message) }
  }

  async function replay(id: string) {
    try { const result = await api.replay(id); changeAutoFollow(false); setEvents(result.events); setActiveEvent(Math.max(result.events.length - 1, 0)) }
    catch (err) { setError((err as Error).message) }
  }

  async function reindex(id: string) {
    setEvents([]); setBusy(true); setError(''); changeAutoFollow(true); resetPlayback()
    try {
      const result = await api.reindex(id, chunkSize, overlap)
      subscribe(`/api/jobs/${result.job_id}/events`, pushEvent, () => { setBusy(false); refresh() }, (message) => { setBusy(false); setError(message) })
      await refresh()
    } catch (err) { setBusy(false); setError((err as Error).message) }
  }

  async function remove(document: DocumentSummary) {
    if (!window.confirm(`Delete ${document.filename} and all of its local vectors?`)) return
    await api.remove(document.id); await refresh()
  }

  const latest = events[activeEvent]
  const selectedChunks = useMemo(() => {
    const event = [...events].reverse().find((item) => item.stage === 'selection' && item.status === 'completed')
    return (event?.payload.selected || []) as RankedChunk[]
  }, [events])
  const points = useMemo(() => {
    const event = [...events].reverse().find((item) => item.stage === 'projection')
    return (event?.payload.points || []) as ProjectionPoint[]
  }, [events])
  const answer = events.filter((event) => event.stage === 'generation' && event.status === 'progress').map((event) => String(event.payload.token || '')).join('')
  const answerMarkdown = useMemo(() => selectedChunks.reduce(
    (markdown, chunk) => markdown.replaceAll(chunk.marker, '[`' + chunk.marker + '`](#source-' + chunk.chunk_id + ')'),
    answer,
  ), [answer, selectedChunks])
  const prompt = [...events].reverse().find((event) => event.stage === 'prompt')?.payload.prompt as string | undefined
  const graphTokenEvent = [...events].reverse().find((event) => event.stage === 'query_tokenization' && event.status === 'completed')
  const graphVectorEvent = [...events].reverse().find((event) => event.stage === 'query_embedding' && event.status === 'completed')
  const graphTokens = (graphTokenEvent?.payload.token_ids || []) as number[]
  const graphVector = (graphVectorEvent?.payload.vector || graphVectorEvent?.payload.vector_preview || []) as number[]
  const graphDimensions = Number(graphVectorEvent?.payload.dimensions || graphVector.length)
  const graphInput = String(graphTokenEvent?.payload.embedding_input || '')
  const rawClsVector = (graphVectorEvent?.payload.raw_cls_vector || []) as number[]
  const contextBudget = [...events].reverse().find((event) => event.stage === 'prompt_budget' && event.status === 'completed')?.payload
  const contextLimit = Number(contextBudget?.context_limit || 0)
  const promptTokens = Number(contextBudget?.prompt_tokens || 0)
  const reservedTokens = Number(contextBudget?.answer_reserve || 0)
  const freeTokens = Math.max(0, contextLimit - promptTokens - reservedTokens)
  const budgetPercent = contextLimit ? Math.min(100, (promptTokens / contextLimit) * 100) : 0
  const reservePercent = contextLimit ? Math.min(100 - budgetPercent, (reservedTokens / contextLimit) * 100) : 0
  const removedForContext = (contextBudget?.removed_for_context || []) as string[]
  const ingestionPoints = ([...events].reverse().find((event) => event.stage === 'embedding' && event.status === 'completed')?.payload.points || []) as ProjectionPoint[]
  const storedRecord = [...events].reverse().find((event) => event.stage === 'storage' && event.status === 'completed')?.payload.stored_record
  const storedRecords = ([...events].reverse().find((event) => event.stage === 'storage' && event.status === 'completed')?.payload.stored_records || (storedRecord ? [storedRecord] : [])) as StoredVectorRecord[]

  const changeWorkflow = (next: 'knowledge' | 'questions') => {
    if (next === workflow) return
    setWorkflow(next)
    setEvents([])
    setActiveEvent(0)
    setError('')
    changeAutoFollow(true)
  }

  return (
    <div className="workspace">
      <section className="workspace-head">
        <div><span className="eyebrow">Module 01 · real pipeline</span><h2>Document RAG explorer</h2><p>Follow your data from file bytes to a grounded local answer.</p></div>
        <div className="truth-label"><span /> Runtime data from your machine</div>
      </section>

      <nav className="workflow-tabs" aria-label="RAG workflows">
        <button className={workflow === 'knowledge' ? 'active' : ''} onClick={() => changeWorkflow('knowledge')} disabled={busy}>
          <UploadCloud size={18} /><span><strong>Add knowledge</strong><small>Upload, chunk, and index documents</small></span>
        </button>
        <button className={workflow === 'questions' ? 'active' : ''} onClick={() => changeWorkflow('questions')} disabled={busy}>
          <Search size={18} /><span><strong>Ask questions</strong><small>Retrieve sources and generate answers</small></span>
        </button>
      </nav>

      {error && <div className="error-banner" role="alert">{error}<button onClick={() => setError('')}>Dismiss</button></div>}
      <div className="rag-layout">
        <aside className="control-panel">
          {workflow === 'knowledge' && <div className="panel-section">
            <div className="section-title"><span>1</span><div><strong>Add knowledge</strong><small>PDF · TXT · MD, up to 20 MB</small></div></div>
            <button
              className={`drop-zone ${dragging ? 'dragging' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); upload(e.dataTransfer.files[0]) }}
              onClick={() => inputRef.current?.click()} disabled={busy}
            >
              <UploadCloud size={26} /><strong>Drop a document</strong><small>or click to choose a file</small>
            </button>
            <input ref={inputRef} hidden type="file" accept=".pdf,.txt,.md,.markdown" onChange={(e) => upload(e.target.files?.[0])} />
            <details className="advanced"><summary><SlidersHorizontal size={14} /> Chunking controls</summary>
              <label>Chunk size <output>{chunkSize} tokens</output><input type="range" min="32" max="1024" step="32" value={chunkSize} onChange={(e) => { const value=Number(e.target.value); setChunkSize(value); if(overlap>=value) setOverlap(Math.max(0,value-32)) }} /></label>
              <label>Overlap <output>{overlap} tokens</output><input type="range" min="0" max={Math.max(0, chunkSize - 1)} step="8" value={overlap} onChange={(e) => setOverlap(Number(e.target.value))} /></label>
            </details>
          </div>}

          <div className="panel-section library">
            <div className="section-title"><span>{workflow === 'knowledge' ? '2' : '1'}</span><div><strong>{workflow === 'knowledge' ? 'Knowledge library' : 'Choose sources'}</strong><small>{documents.length} local document{documents.length === 1 ? '' : 's'}</small></div></div>
            {!documents.length && <div className="quiet-empty"><BookOpen size={20} />Your indexed documents will appear here.</div>}
            {documents.map((doc) => <article className={`document-row ${selected.includes(doc.id) ? 'selected' : ''}`} key={doc.id}>
              <button className="document-select" disabled={doc.status !== 'ready'} onClick={() => setSelected((value) => value.includes(doc.id) ? value.filter((id) => id !== doc.id) : [...value, doc.id])}>
                <span className="file-icon"><FileText size={16} /></span><span><strong>{doc.filename}</strong><small>{doc.status === 'ready' ? `${doc.chunk_count} chunks · ${doc.token_count} tokens` : doc.status}</small></span>
              </button>
              <div className="row-actions"><button title="Replay pipeline" onClick={() => replay(doc.id)}><Play size={13} /></button><button title="Reindex" onClick={() => reindex(doc.id)}><RotateCcw size={13} /></button><button title="Delete" onClick={() => remove(doc)}><Trash2 size={13} /></button></div>
              {doc.reindex_required && <span className="stale">Model changed · reindex</span>}
            </article>)}
          </div>

          {workflow === 'questions' && <div className="panel-section">
            <div className="section-title"><span>2</span><div><strong>Ask your library</strong><small>Search, inspect, then generate</small></div></div>
            <textarea value={query} onChange={(e) => setQuery(e.target.value)} rows={4} />
            <div className="inline-controls"><label>Top K<input type="number" min="1" max="20" value={topK} onChange={(e) => setTopK(Number(e.target.value))} /></label><label>Min score<input type="number" min="-1" max="1" step="0.05" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} /></label></div>
            <label className="switch"><input type="checkbox" checked={generate} onChange={(e) => setGenerate(e.target.checked)} /><span />Generate a cited answer</label>
            <button className="primary-button" onClick={runSearch} disabled={busy || !query.trim()}><Search size={17} />{busy ? 'Pipeline running…' : 'Run the pipeline'}</button>
          </div>}
        </aside>

        <main className="process-canvas">
          <div className="process-top"><div><span className="eyebrow">Live execution trace</span><h3>{workflow === 'knowledge' ? 'Knowledge ingestion' : 'Question answering'}</h3></div><div className="trace-actions">
            <div className="playback-mode" aria-label="Pipeline playback mode">
              <button className={stepMode ? 'active' : ''} onClick={enableStepMode}><PauseCircle size={13} />Step mode</button>
              <button className={!stepMode ? 'active' : ''} onClick={resumeLive}><PlayCircle size={13} />Live</button>
            </div>
            {stepMode && playbackPaused && <button className="next-phase" onClick={revealNextPhase}><StepForward size={13} />Next phase{bufferedCount ? ` · ${bufferedCount} buffered` : ''}</button>}
            {events.length > 0 && <span className="event-count">{events.length} visible</span>}
            {!autoFollow && <button className="resume-live" onClick={() => { changeAutoFollow(true); setActiveEvent(Math.max(events.length - 1, 0)) }}>Jump to latest</button>}
          </div></div>
          <div className="process-grid">
            <PipelineTimeline events={events} active={activeEvent} onSelect={inspectEvent} />
            <EventInspector event={latest} events={events} activeIndex={activeEvent} onSelect={inspectEvent} />
          </div>
          {workflow === 'knowledge' && ingestionPoints.length > 0 && <section className="ingestion-artifact">
            <header><div><span className="eyebrow">Embedding output</span><h3>Vector space projection</h3></div><small>PCA-style 2D projection of real stored vectors</small></header>
            <VectorPlot points={ingestionPoints} />
          </section>}
          {workflow === 'knowledge' && storedRecords.length > 0 && <section className="stored-record">
            <header><div><span className="eyebrow">Vector database</span><h3>All stored vector records</h3></div><small>{storedRecords.length} records · scroll to inspect more</small></header>
            <VectorStoreBrowser records={storedRecords} />
          </section>}
          {graphTokens.length > 0 && graphVector.length > 0 && <TokenEmbeddingGraph
            tokenIds={graphTokens} input={graphInput} vector={graphVector} dimensions={graphDimensions}
            rawClsVector={rawClsVector} rawClsNorm={Number(graphVectorEvent?.payload.raw_cls_norm || 0)}
            returnedNorm={Number(graphVectorEvent?.payload.returned_vector_norm || 0)} finalNorm={Number(graphVectorEvent?.payload.final_vector_norm || 0)}
          />}
          {contextLimit > 0 && <section className="context-budget" aria-label="Chat context budget">
            <header><div><span className="eyebrow">Actual formatted prompt</span><h3>Context-window budget</h3></div><strong>{promptTokens + reservedTokens} / {contextLimit} tokens reserved</strong></header>
            <div className="budget-track" aria-hidden="true"><span className="budget-prompt" style={{ width: `${budgetPercent}%` }} /><span className="budget-reserve" style={{ width: `${reservePercent}%` }} /></div>
            <div className="budget-legend"><span><i className="prompt-dot" />Prompt <strong>{promptTokens}</strong></span><span><i className="reserve-dot" />Answer reserve <strong>{reservedTokens}</strong></span><span><i className="free-dot" />Free <strong>{freeTokens}</strong></span></div>
            {removedForContext.length > 0 && <p>Removed to fit the model context: {removedForContext.map((marker) => <code key={marker}>{marker}</code>)}</p>}
          </section>}
          {selectedChunks.length > 0 && <section className="results"><header><span className="eyebrow">Retrieved context</span><h3>Why these chunks?</h3></header><div className="result-layout"><div className="chunk-list">{selectedChunks.map((chunk) => <article className="chunk-card" id={`source-${chunk.chunk_id}`} key={chunk.chunk_id}><div><code>{chunk.marker}</code><strong>{chunk.filename} · page {chunk.page}</strong><output>{(chunk.score * 100).toFixed(1)}%</output></div><div className="score-track"><span style={{ width: `${Math.max(0, chunk.score) * 100}%` }} /></div><p>{chunk.text}</p></article>)}</div><VectorPlot points={points} /></div></section>}
          {prompt && <details className="prompt-preview"><summary>See the exact prompt sent to llama.cpp</summary><pre>{prompt}</pre></details>}
          {answer && <section className="answer-card"><span className="eyebrow">Local model answer</span><div className="markdown-answer"><ReactMarkdown remarkPlugins={[remarkGfm]}>{answerMarkdown}</ReactMarkdown></div><small>Only source markers returned by the selected context are treated as valid citations.</small></section>}
        </main>
      </div>
    </div>
  )
}
