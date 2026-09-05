import type { ProjectionPoint } from '../types'

export function VectorPlot({ points }: { points: ProjectionPoint[] }) {
  if (!points.length) return null
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y)
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys)
  const place = (value: number, min: number, max: number) => max === min ? 50 : 8 + ((value - min) / (max - min)) * 84
  return (
    <div className="vector-plot" aria-label="Two dimensional projection of query and candidate vectors">
      <span className="axis x" /><span className="axis y" />
      {points.map((point) => (
        point.chunk_id
          ? <a href={`#source-${point.chunk_id}`} title={`${point.filename} · ${((point.score || 0) * 100).toFixed(1)}%`} key={point.id} className="vector-point" style={{ left: `${place(point.x, minX, maxX)}%`, bottom: `${place(point.y, minY, maxY)}%` }}><i /><small>{point.id}</small></a>
          : <span key={point.id} className="vector-point query" style={{ left: `${place(point.x, minX, maxX)}%`, bottom: `${place(point.y, minY, maxY)}%` }}><i /><small>{point.id}</small></span>
      ))}
    </div>
  )
}
