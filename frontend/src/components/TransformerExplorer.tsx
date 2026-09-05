import { useEffect, useMemo, useState } from 'react'
import { ChevronLeft, ChevronRight, FlaskConical, Pause, Play, RotateCcw, Send } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api, subscribe } from '../api'
import type { StageEvent } from '../types'
import { EventInspector } from './EventInspector'

type LessonStep = { title: string; subtitle: string; formula: string; explanation: string; kind: string }

const lesson: LessonStep[] = [
  { title: 'Token + position', subtitle: 'Give every token identity and order', formula: 'xᵢ = E[tokenᵢ] + P[i]', explanation: 'The token lookup says what a token is. Its position vector says where it appeared. The decoder begins by adding them.', kind: 'vectors' },
  { title: 'Project Q, K and V', subtitle: 'Create three views of each token', formula: 'Q = XWQ  ·  K = XWK  ·  V = XWV', explanation: 'Queries describe what a token is looking for; keys describe what it offers; values carry the information that can be moved.', kind: 'qkv' },
  { title: 'Score relationships', subtitle: 'Compare each query with every allowed key', formula: 'S = QKᵀ / √dₖ', explanation: 'A dot product becomes large when a query and key point in similar directions. Scaling keeps softmax stable.', kind: 'scores' },
  { title: 'Apply causal mask', subtitle: 'The future is hidden', formula: 'Sᵢⱼ = −∞ when j > i', explanation: 'During generation, a position may only use itself and earlier positions. Future cells are removed before softmax.', kind: 'mask' },
  { title: 'Normalize attention', subtitle: 'Turn scores into routing weights', formula: 'A = softmax(S)', explanation: 'Each row becomes positive and sums to one. Brighter cells receive more of the destination token’s attention.', kind: 'softmax' },
  { title: 'Move information', subtitle: 'Mix value vectors using the weights', formula: 'H = AV', explanation: 'The weighted values are combined for each token. Multiple heads repeat this with different learned projections.', kind: 'attention' },
  { title: 'Residual + MLP', subtitle: 'Preserve, normalize and transform', formula: 'y = x + Attn(LN(x))\nz = y + MLP(LN(y))', explanation: 'Residual paths preserve earlier information while the feed-forward network transforms each token independently.', kind: 'mlp' },
  { title: 'Choose the next token', subtitle: 'Project the final state to vocabulary scores', formula: 'p(token) = softmax(hWᵀ / T)', explanation: 'Temperature reshapes probabilities. A sampling rule selects one token, appends it, and runs the decoder again.', kind: 'sampling' },
]

function seededMatrix(seedText: string, masked: boolean, normalized: boolean) {
  let seed = [...seedText].reduce((sum, char) => (sum * 31 + char.charCodeAt(0)) >>> 0, 7)
  const size = 5
  const values = Array.from({ length: size }, (_, row) => Array.from({ length: size }, (_, col) => {
    seed = (seed * 1664525 + 1013904223) >>> 0
    if (masked && col > row) return normalized ? 0 : -9
    return ((seed % 160) - 50) / 100
  }))
  if (!normalized) return values
  return values.map((row, rowIndex) => {
    const allowed = row.slice(0, rowIndex + 1)
    const max = Math.max(...allowed)
    const exp = allowed.map((value) => Math.exp(value - max))
    const total = exp.reduce((a, b) => a + b, 0)
    return row.map((_, col) => col <= rowIndex ? exp[col] / total : 0)
  })
}

function Simulation({ step, prompt }: { step: LessonStep; prompt: string }) {
  const normalized = ['softmax', 'attention', 'mlp', 'sampling'].includes(step.kind)
  const masked = ['mask', 'softmax', 'attention', 'mlp', 'sampling'].includes(step.kind)
  const matrix = useMemo(() => seededMatrix(prompt, masked, normalized), [prompt, masked, normalized])
  const tokens = prompt.trim().split(/\s+/).slice(0, 5)
  while (tokens.length < 5) tokens.push(['the', 'model', 'sees', 'a', 'token'][tokens.length])
  return (
    <div className={`simulation ${step.kind}`}>
      <div className="simulation-label"><FlaskConical size={14} /> Educational miniature · not GGUF activations</div>
      <div className="token-row">{tokens.map((token, index) => <span key={`${token}-${index}`} style={{ '--token': index } as React.CSSProperties}>{token}</span>)}</div>
      {step.kind === 'vectors' || step.kind === 'qkv' ? <div className="vector-stack">{tokens.map((token, row) => <div key={token}><small>{step.kind === 'qkv' ? ['Q', 'K', 'V'][row % 3] : `x${row + 1}`}</small>{matrix[row].map((value, col) => <i key={col} style={{ height: `${22 + Math.abs(value) * 55}px`, opacity: .45 + Math.abs(value) / 2 }} />)}</div>)}</div> : <div className="matrix"><span />{tokens.map((token) => <small key={token}>{token.slice(0, 4)}</small>)}{matrix.map((row, rowIndex) => [<small key={`r-${rowIndex}`}>{tokens[rowIndex].slice(0, 4)}</small>, ...row.map((value, col) => <i key={`${rowIndex}-${col}`} className={value === -9 ? 'blocked' : ''} style={{ '--heat': Math.max(0, value) } as React.CSSProperties} title={value === -9 ? 'masked' : value.toFixed(3)}>{value === -9 ? '×' : value.toFixed(2)}</i>)])}</div>}
    </div>
  )
}

export function TransformerExplorer() {
  const [prompt, setPrompt] = useState('Local models make private AI possible.')
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [mode, setMode] = useState<'guided' | 'sandbox'>('guided')
  const [events, setEvents] = useState<StageEvent[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const current = lesson[step]
  useEffect(() => {
    if (!playing) return
    const timer = window.setInterval(() => setStep((value) => {
      if (value === lesson.length - 1) { setPlaying(false); return value }
      return value + 1
    }), 1800)
    return () => window.clearInterval(timer)
  }, [playing])

  async function runModel() {
    setError(''); setEvents([]); setBusy(true)
    try {
      const result = await api.transformer({ prompt, max_tokens: 64, temperature: 0.7 })
      subscribe(`/api/transformer/runs/${result.run_id}/events`, (event) => setEvents((value) => [...value, event]), () => setBusy(false), (message) => { setBusy(false); setError(message) })
    } catch (err) { setBusy(false); setError((err as Error).message) }
  }
  const output = events.filter((event) => event.stage === 'generation' && event.status === 'progress').map((event) => event.payload.token).join('')
  const tokenEvent = events.find((event) => event.stage === 'tokenization')
  return (
    <div className="workspace transformer-workspace">
      <section className="workspace-head"><div><span className="eyebrow">Module 02 · model + simulation</span><h2>Transformer explorer</h2><p>Keep observable llama.cpp facts separate from the math lesson.</p></div><div className="segmented"><button className={mode === 'guided' ? 'active' : ''} onClick={() => setMode('guided')}>Guided</button><button className={mode === 'sandbox' ? 'active' : ''} onClick={() => setMode('sandbox')}>Sandbox</button></div></section>
      {error && <div className="error-banner">{error}</div>}
      <div className="transformer-layout">
        <main className="lesson-card">
          <header><div><span className="step-number">{String(step + 1).padStart(2, '0')}</span><small> of {lesson.length}</small><h3>{current.title}</h3><p>{current.subtitle}</p></div><code className="formula">{current.formula}</code></header>
          <Simulation step={current} prompt={prompt} />
          <p className="lesson-copy">{current.explanation}</p>
          <footer><div className="lesson-progress">{lesson.map((item, index) => <button key={item.title} className={index === step ? 'active' : index < step ? 'seen' : ''} onClick={() => setStep(index)} aria-label={`Open ${item.title}`} />)}</div><div className="playback"><button onClick={() => setStep(Math.max(0, step - 1))} aria-label="Previous step"><ChevronLeft /></button><button className="play" onClick={() => setPlaying(!playing)} aria-label={playing ? 'Pause' : 'Play'}>{playing ? <Pause /> : <Play />}</button><button onClick={() => setStep(Math.min(lesson.length - 1, step + 1))} aria-label="Next step"><ChevronRight /></button><button onClick={() => { setStep(0); setPlaying(false) }} aria-label="Restart"><RotateCcw size={16} /></button></div></footer>
        </main>
        <aside className="runtime-card">
          <div><span className="truth-label"><span /> Observable llama.cpp data</span><h3>What your model did</h3><p>These values come directly from the local HTTP server.</p></div>
          <label>Prompt<textarea rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)} /></label>
          <button className="primary-button" disabled={busy || !prompt.trim()} onClick={runModel}><Send size={16} />{busy ? 'Generating locally…' : 'Run on llama.cpp'}</button>
          {tokenEvent && <div className="runtime-stat"><small>Input token IDs</small><div className="token-ids">{(tokenEvent.payload.token_ids as number[]).map((token, index) => <code key={`${token}-${index}`}>{token}</code>)}</div></div>}
          {output && <div className="model-output"><small>Streamed output</small><div className="markdown-answer"><ReactMarkdown remarkPlugins={[remarkGfm]}>{output}</ReactMarkdown></div></div>}
          {events.length > 0 && <details><summary>Inspect latest raw event</summary><EventInspector event={events.at(-1)} /></details>}
          <div className="boundary-note"><strong>Why no real attention matrix?</strong><p>llama-server does not expose stable internal activations. The lesson at left uses a tiny deterministic simulation and is deliberately labeled.</p></div>
        </aside>
      </div>
    </div>
  )
}
