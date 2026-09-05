// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { StageEvent } from '../types'
import { EventInspector } from './EventInspector'
import { RagExplorer } from './RagExplorer'
import { VectorPlot } from './VectorPlot'
import { VectorDimensions, VectorStoreBrowser } from './VectorStoreBrowser'

const mocks = vi.hoisted(() => ({
  documents: vi.fn().mockResolvedValue([]),
  upload: vi.fn().mockResolvedValue({ document_id: 'doc-1', job_id: 'job-1' }),
  subscribe: vi.fn(),
}))

vi.mock('../api', () => ({
  api: { documents: mocks.documents, upload: mocks.upload },
  subscribe: mocks.subscribe,
}))

const events: StageEvent[] = [
  { stage: 'generation', status: 'started', sequence: 1, timestamp: '', payload: {} },
  { stage: 'generation', status: 'progress', sequence: 2, timestamp: '', payload: { token: 'Hello' } },
  { stage: 'generation', status: 'completed', sequence: 3, timestamp: '', payload: { answer: '**Hello**' } },
]

afterEach(cleanup)

describe('RAG visualization', () => {
  it('separates adding knowledge from asking questions', async () => {
    render(<RagExplorer />)

    expect(screen.getByText('Drop a document')).toBeInTheDocument()
    expect(screen.queryByRole('textbox')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /Ask questions/ }))

    expect(screen.getByRole('textbox')).toBeInTheDocument()
    expect(screen.queryByText('Drop a document')).not.toBeInTheDocument()
  })

  it('pauses completed ingestion phases until the user reveals the next phase', async () => {
    let onEvent: ((event: StageEvent) => void) | undefined
    mocks.subscribe.mockImplementation((_url, handler) => { onEvent = handler })
    const view = render(<RagExplorer />)
    const input = view.container.querySelector('input[type="file"]') as HTMLInputElement

    fireEvent.change(input, { target: { files: [new File(['hello'], 'notes.txt', { type: 'text/plain' })] } })
    await waitFor(() => expect(mocks.subscribe).toHaveBeenCalled())

    act(() => {
      onEvent?.({ stage: 'upload', status: 'completed', sequence: 1, timestamp: '', payload: { input: {}, process: 'Upload', output: {} } })
      onEvent?.({ stage: 'validation', status: 'started', sequence: 2, timestamp: '', payload: {} })
      onEvent?.({ stage: 'validation', status: 'completed', sequence: 3, timestamp: '', payload: { input: {}, process: 'Validate', output: {} } })
    })

    expect(screen.getByText('1 visible')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Next phase · 2 buffered/ })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /Next phase/ }))
    expect(screen.getByText('3 visible')).toBeInTheDocument()
  })

  it('allows inspection of every grouped stage update', () => {
    const select = vi.fn()
    render(<EventInspector event={events[2]} events={events} activeIndex={2} onSelect={select} />)
    expect(screen.getByText('Update 3 of 3')).toBeInTheDocument()
    fireEvent.click(screen.getByLabelText('Previous stage update'))
    expect(select).toHaveBeenCalledWith(1)
  })

  it('places long answer content in a full-width scrollable payload', () => {
    const view = render(<EventInspector event={{ stage: 'complete', status: 'completed', sequence: 4, timestamp: '', payload: { duration_ms: 10, result_count: 5, answer: 'A'.repeat(500) } }} />)
    const answerLabel = view.container.querySelector('.payload-wide > small')
    expect(answerLabel?.closest('.payload')).toHaveClass('payload-wide')
  })

  it('links projected chunks to their source cards', () => {
    render(<VectorPlot points={[
      { id: 'query', x: 0, y: 0 },
      { id: '[D1:P1:C1]', x: 1, y: 1, chunk_id: 'chunk-1', filename: 'notes.txt', score: 0.9 },
    ]} />)
    expect(screen.getByRole('link', { name: '[D1:P1:C1]' })).toHaveAttribute('href', '#source-chunk-1')
  })

  it('provides a virtualized scroll region for every stored vector record', () => {
    const records = Array.from({ length: 100 }, (_, index) => ({
      id: `chunk-${index}`, document_id: 'doc-1', chunk_index: index, page: 1,
      text: `Stored content ${index}`, token_count: 12, embedding_dimensions: 384,
      vector: [0.1, 0.2, 0.3],
    }))
    render(<VectorStoreBrowser records={records} />)
    expect(screen.getByLabelText('All 100 stored vector records')).toBeInTheDocument()
    expect(document.querySelectorAll('.stored-vector-row').length).toBeLessThan(100)
  })

  it('exposes the complete d0-to-final range in a horizontal virtualizer', () => {
    render(<VectorDimensions vector={[0.1, 0.2, 0.3]} />)
    expect(screen.getByLabelText('Vector dimensions d0 through d2')).toBeInTheDocument()
  })
})
