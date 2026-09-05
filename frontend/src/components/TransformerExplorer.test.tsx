// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { TransformerExplorer } from './TransformerExplorer'

describe('TransformerExplorer', () => {
  it('labels simulated activations and navigates the lesson', () => {
    render(<TransformerExplorer />)
    expect(screen.getByText(/not GGUF activations/i)).toBeInTheDocument()
    expect(screen.getByText('Token + position')).toBeInTheDocument()
    fireEvent.click(screen.getByLabelText('Next step'))
    expect(screen.getByText('Project Q, K and V')).toBeInTheDocument()
  })
})

