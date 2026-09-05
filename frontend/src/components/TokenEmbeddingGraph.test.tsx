// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api'
import { TokenEmbeddingGraph } from './TokenEmbeddingGraph'

vi.mock('@tanstack/react-virtual', () => ({
  useVirtualizer: ({ count, horizontal }: { count: number; horizontal?: boolean }) => ({
    getTotalSize: () => count * (horizontal ? 88 : 37),
    getVirtualItems: () => Array.from({ length: count }, (_, index) => ({ key: index, index, start: index * (horizontal ? 88 : 37), size: horizontal ? 88 : 37 })),
  }),
}))

describe('token embedding graph', () => {
  beforeEach(() => vi.restoreAllMocks())

  it('loads the real contextual vector only after a token is clicked', async () => {
    const contextual = vi.spyOn(api, 'contextualToken').mockResolvedValue({
      token_index: 1, token_id: 22, token_piece: ' token', token_count: 2, dimensions: 4,
      vector: [0.1, 0.2, 0.3, 0.4], server_url: 'http://192.168.31.92:8082', pooling: 'none', normalized: false,
    })
    render(<TokenEmbeddingGraph tokenIds={[11, 22]} input="a token" vector={[0.5, 0.5, 0.5, 0.5]} dimensions={4} finalNorm={1} />)
    expect(contextual).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Inspect token 1, ID 22' }))
    await waitFor(() => expect(contextual).toHaveBeenCalledWith('a token', 1, [11, 22]))
    expect(await screen.findByText('" token"')).toBeInTheDocument()
    expect(screen.getByLabelText('Contextual vector for token 1, 4 dimensions')).toBeInTheDocument()
  })
})
