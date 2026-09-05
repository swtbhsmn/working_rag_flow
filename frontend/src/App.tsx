import { useEffect, useState } from 'react'
import { Boxes, Github, Library, Sparkles } from 'lucide-react'
import { api } from './api'
import type { SystemModels } from './types'
import { ModelStatus } from './components/ModelStatus'
import { RagExplorer } from './components/RagExplorer'
import { TransformerExplorer } from './components/TransformerExplorer'

type Module = 'rag' | 'transformer'

export default function App() {
  const [module, setModule] = useState<Module>('rag')
  const [models, setModels] = useState<SystemModels>()
  const [loading, setLoading] = useState(true)
  const loadModels = () => { setLoading(true); api.models().then(setModels).catch(() => setModels(undefined)).finally(() => setLoading(false)) }
  useEffect(loadModels, [])
  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Under the Token home"><span><Sparkles size={17} /></span><strong>Under the Token</strong><small>LOCAL AI LAB</small></a>
        <nav aria-label="Learning modules"><button className={module === 'rag' ? 'active' : ''} onClick={() => setModule('rag')}><Library size={16} />RAG pipeline</button><button className={module === 'transformer' ? 'active' : ''} onClick={() => setModule('transformer')}><Boxes size={16} />Transformer</button></nav>
        <a className="github-link" href="https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md" target="_blank" rel="noreferrer"><Github size={16} />llama.cpp API</a>
      </header>
      <div className="hero" id="top"><div><span className="eyebrow">No cloud. No mystery.</span><h1>Watch local AI<br /><em>think in stages.</em></h1></div><p>Upload a document or send a prompt. Every visible runtime value comes from your machine; every simulation tells you exactly what it is.</p></div>
      <ModelStatus models={models} loading={loading} onRefresh={loadModels} />
      {module === 'rag' ? <RagExplorer /> : <TransformerExplorer />}
      <footer className="site-footer"><span>Runs locally through React, FastAPI, SQLite and llama.cpp.</span><span>Your documents never leave this machine.</span></footer>
    </div>
  )
}
