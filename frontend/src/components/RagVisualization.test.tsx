// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { StageEvent } from '../types'
import { EventInspector } from './EventInspector'
import { VectorPlot } from './VectorPlot'

const events: StageEvent[] = [
  { stage: 'generation', status: 'started', sequence: 1, timestamp: '', payload: {} },
  { stage: 'generation', status: 'progress', sequence: 2, timestamp: '', payload: { token: 'Hello' } },
  { stage: 'generation', status: 'completed', sequence: 3, timestamp: '', payload: { answer: '**Hello**' } },
]

describe('RAG visualization', () => {
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
})
