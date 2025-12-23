import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

type Role = 'GK' | 'DEF'

type Player = {
  id: number
  x: number // meters from goal line (0 = goal line, positive into pitch)
  y: number // meters from goal centre line (positive to the right)
  role: Role
}

type OpenGoalResult = {
  fractionOpen: number
  idealTargetY: number | null
}

// Pitch & goal dimensions (in meters)
const PITCH_LENGTH = 52.5 // half pitch, standard 105m total
const PITCH_WIDTH = 68
const GOAL_WIDTH = 7.32

// Model tuning constants (from our "common sense" calibration)
const K_DISTANCE = 35
const BETA_OPEN = 1.2
const A_DEF_BETWEEN = 0.4
const B_PRESSURE = 0.8

const DEFENDER_RADIUS = 0.6 // effective blocking radius (m)
const GK_RADIUS = 0.8

function distance(x1: number, y1: number, x2: number, y2: number) {
  const dx = x2 - x1
  const dy = y2 - y1
  return Math.hypot(dx, dy)
}

function computeAngleToPosts(shotX: number, shotY: number): number {
  const leftPostY = -GOAL_WIDTH / 2
  const rightPostY = GOAL_WIDTH / 2

  const thetaL = Math.atan2(leftPostY - shotY, 0 - shotX)
  const thetaR = Math.atan2(rightPostY - shotY, 0 - shotX)

  return Math.abs(thetaR - thetaL)
}

function computeBaseLocationTerm(d: number, alpha: number): number {
  const safeD = Math.max(d, 0.5)

  const baseCentral = Math.min(0.95, K_DISTANCE / (safeD * safeD))

  const alphaCentral = 2 * Math.atan((GOAL_WIDTH / 2) / safeD)
  const effectiveAlpha = Math.min(alpha, alphaCentral)

  if (alphaCentral <= 0) return 0

  const ratio = Math.max(effectiveAlpha / alphaCentral, 0)
  const angleFactor = Math.pow(ratio, 1.5) // gamma = 1.5

  return baseCentral * angleFactor
}

// Compute open goal fraction and the ideal (largest open) target point on the goal line
function computeOpenGoal(
  shotX: number,
  shotY: number,
  players: Player[],
  samples = 40,
): OpenGoalResult {
  const sampleYs: number[] = []
  const step = GOAL_WIDTH / samples
  const yStart = -GOAL_WIDTH / 2 + step / 2

  for (let i = 0; i < samples; i++) {
    sampleYs.push(yStart + i * step)
  }

  const isOpen: boolean[] = new Array(samples).fill(true)

  for (let j = 0; j < samples; j++) {
    const gy = sampleYs[j]
    const gx = 0
    const vx = gx - shotX
    const vy = gy - shotY
    const vLen2 = vx * vx + vy * vy

    if (vLen2 === 0) {
      isOpen[j] = false
      continue
    }

    for (const p of players) {
      const radius = p.role === 'GK' ? GK_RADIUS : DEFENDER_RADIUS
      const wx = p.x - shotX
      const wy = p.y - shotY
      const t = (wx * vx + wy * vy) / vLen2
      if (t <= 0 || t >= 1) continue

      const projX = shotX + t * vx
      const projY = shotY + t * vy
      const dPerp = distance(p.x, p.y, projX, projY)

      if (dPerp <= radius) {
        isOpen[j] = false
        break
      }
    }
  }

  const openIndices: number[] = []
  for (let j = 0; j < samples; j++) {
    if (isOpen[j]) openIndices.push(j)
  }

  const fractionOpen = openIndices.length / samples

  if (openIndices.length === 0) {
    return {
      fractionOpen,
      idealTargetY: null,
    }
  }

  // Find contiguous open segments and pick the one with largest angular width
  type Segment = { start: number; end: number }
  const segments: Segment[] = []
  let currentStart = openIndices[0]
  let prev = openIndices[0]

  for (let i = 1; i < openIndices.length; i++) {
    const idx = openIndices[i]
    if (idx === prev + 1) {
      prev = idx
    } else {
      segments.push({ start: currentStart, end: prev })
      currentStart = idx
      prev = idx
    }
  }
  segments.push({ start: currentStart, end: prev })

  let bestSegment: Segment | null = null
  let bestAngularWidth = -Infinity

  for (const seg of segments) {
    const yStartSeg = sampleYs[seg.start]
    const yEndSeg = sampleYs[seg.end]
    const thetaStart = Math.atan2(yStartSeg - shotY, 0 - shotX)
    const thetaEnd = Math.atan2(yEndSeg - shotY, 0 - shotX)
    const width = Math.abs(thetaEnd - thetaStart)

    if (width > bestAngularWidth) {
      bestAngularWidth = width
      bestSegment = seg
    }
  }

  if (!bestSegment) {
    return {
      fractionOpen,
      idealTargetY: null,
    }
  }

  const midIndex = Math.round((bestSegment.start + bestSegment.end) / 2)
  const idealTargetY = sampleYs[midIndex]

  return {
    fractionOpen,
    idealTargetY,
  }
}

function computeDefenderMetrics(
  shotX: number,
  shotY: number,
  players: Player[],
): { nBetween: number; minDefenderDist: number } {
  const defenders = players.filter((p) => p.role !== 'GK')
  if (defenders.length === 0) {
    return { nBetween: 0, minDefenderDist: 30 }
  }

  const goalCenterX = 0
  const goalCenterY = 0
  const gx = goalCenterX - shotX
  const gy = goalCenterY - shotY
  const vLen2 = gx * gx + gy * gy

  let nBetween = 0
  let minDist = Infinity

  for (const d of defenders) {
    const dx = d.x - shotX
    const dy = d.y - shotY
    const dist = Math.hypot(dx, dy)
    if (dist < minDist) minDist = dist

    if (vLen2 === 0) continue

    const t = (dx * gx + dy * gy) / vLen2
    if (t <= 0 || t >= 1) continue

    const projX = shotX + t * gx
    const projY = shotY + t * gy
    const lateral = distance(d.x, d.y, projX, projY)

    // Treat defender as "between" if close to shooting line
    if (lateral < 1.5) {
      nBetween += 1
    }
  }

  if (!Number.isFinite(minDist)) minDist = 30

  return { nBetween, minDefenderDist: minDist }
}

function computeGKCoverage(
  shotX: number,
  shotY: number,
  players: Player[],
  idealTargetY: number | null,
): number {
  const gk = players.find((p) => p.role === 'GK')
  if (!gk) return 0.5 // neutral baseline if GK missing

  const targetY = idealTargetY ?? 0 // fall back to goal center

  const sx = shotX
  const sy = shotY
  const tx = 0
  const ty = targetY
  const kx = gk.x
  const ky = gk.y

  const vx = tx - sx
  const vy = ty - sy
  const ux = kx - sx
  const uy = ky - sy

  const vLen2 = vx * vx + vy * vy
  if (vLen2 === 0) return 0.5

  let tGK = (ux * vx + uy * vy) / vLen2
  let tClamped = tGK
  if (tClamped < 0) tClamped = 0
  if (tClamped > 1) tClamped = 1

  const projX = sx + tClamped * vx
  const projY = sy + tClamped * vy
  const dPerp = distance(kx, ky, projX, projY)

  const sigma = 1 // metres
  const cAlign = Math.exp(-dPerp / sigma)
  const cPos = tClamped

  return cPos * cAlign
}

function computeXG(shotX: number, shotY: number, players: Player[]): number {
  if (shotX <= 0) return 0

  const d = distance(shotX, shotY, 0, 0)
  const alpha = computeAngleToPosts(shotX, shotY)

  if (!Number.isFinite(d) || d <= 0) return 0

  const baseLoc = computeBaseLocationTerm(d, alpha)

  const openGoal = computeOpenGoal(shotX, shotY, players)
  const fOpen = openGoal.fractionOpen

  const F_open = fOpen <= 0 ? 0 : Math.pow(fOpen, BETA_OPEN)

  const { nBetween, minDefenderDist } = computeDefenderMetrics(
    shotX,
    shotY,
    players,
  )

  const pressure = Math.max(0, (2 - minDefenderDist) / 2)
  const F_def =
    1 / (1 + A_DEF_BETWEEN * nBetween + B_PRESSURE * pressure || 1)

  const C_gk = computeGKCoverage(shotX, shotY, players, openGoal.idealTargetY)
  const F_gk = 0.4 + 0.8 * (1 - C_gk)

  let xg = baseLoc * F_open * F_def * F_gk
  if (!Number.isFinite(xg)) xg = 0

  xg = Math.max(0, Math.min(0.95, xg))
  return xg
}

function initialPlayers(): Player[] {
  // Simple 4-4-2 defensive shape on a half-pitch
  return [
    { id: 1, role: 'GK', x: 2, y: 0 },
    // Back four
    { id: 2, role: 'DEF', x: 8, y: -18 },
    { id: 3, role: 'DEF', x: 8, y: -6 },
    { id: 4, role: 'DEF', x: 8, y: 6 },
    { id: 5, role: 'DEF', x: 8, y: 18 },
    // Midfield four
    { id: 6, role: 'DEF', x: 20, y: -20 },
    { id: 7, role: 'DEF', x: 20, y: -5 },
    { id: 8, role: 'DEF', x: 20, y: 5 },
    { id: 9, role: 'DEF', x: 20, y: 20 },
    // Two forwards tracking back
    { id: 10, role: 'DEF', x: 32, y: -10 },
    { id: 11, role: 'DEF', x: 32, y: 10 },
  ]
}

function mapPitchToCanvas(
  x: number,
  y: number,
  width: number,
  height: number,
): { px: number; py: number } {
  const px = (x / PITCH_LENGTH) * width
  const py = height / 2 + (y / PITCH_WIDTH) * height
  return { px, py }
}

function mapCanvasToPitch(
  px: number,
  py: number,
  width: number,
  height: number,
): { x: number; y: number } {
  const x = (px / width) * PITCH_LENGTH
  const y = ((py - height / 2) / height) * PITCH_WIDTH
  return { x, y }
}

function xgToColor(xg: number): string {
  const maxXGForColor = 0.5
  const v = Math.max(0, Math.min(xg / maxXGForColor, 1))
  const hue = (1 - v) * 220 // from blue (low) to red (high)
  const saturation = 80
  const lightness = 50
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`
}

function App() {
  const [players, setPlayers] = useState<Player[]>(() => initialPlayers())
  const [hoverXG, setHoverXG] = useState<number | null>(null)
  const [hoverPos, setHoverPos] = useState<{ x: number; y: number } | null>(
    null,
  )

  const [draggingId, setDraggingId] = useState<number | null>(null)
  const pitchRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  const playersMemo = useMemo(() => players, [players])

  // Draw heatmap and pitch markings
  useEffect(() => {
    const canvas = canvasRef.current
    const container = pitchRef.current
    if (!canvas || !container) return

    const rect = container.getBoundingClientRect()

    // Keep the internal canvas resolution in sync with the displayed size
    const displayWidth = Math.max(1, Math.floor(rect.width))
    const displayHeight = Math.max(1, Math.floor(rect.height))
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth
      canvas.height = displayHeight
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const w = canvas.width
    const h = canvas.height

    ctx.clearRect(0, 0, w, h)

    // Background grass
    ctx.fillStyle = '#0b5d2a'
    ctx.fillRect(0, 0, w, h)

    // xG heatmap (coarse grid)
    const cellsX = 70
    const cellsY = 40
    const cellWidth = w / cellsX
    const cellHeight = h / cellsY

    for (let ix = 0; ix < cellsX; ix++) {
      for (let iy = 0; iy < cellsY; iy++) {
        const centerPx = (ix + 0.5) * cellWidth
        const centerPy = (iy + 0.5) * cellHeight
        const { x, y } = mapCanvasToPitch(centerPx, centerPy, w, h)

        // Only consider shots in front of the goal within the half-pitch
        if (x < 0 || x > PITCH_LENGTH) continue
        if (Math.abs(y) > PITCH_WIDTH / 2) continue

        const xg = computeXG(x, y, playersMemo)
        if (xg <= 0) continue

        ctx.fillStyle = xgToColor(xg)
        ctx.fillRect(ix * cellWidth, iy * cellHeight, cellWidth, cellHeight)
      }
    }

    // Draw pitch lines on top
    ctx.strokeStyle = 'rgba(255,255,255,0.9)'
    ctx.lineWidth = 2
    ctx.beginPath()
    // Outer box
    ctx.rect(1, 1, w - 2, h - 2)
    ctx.stroke()

    // Goal line at x = 0
    const goalTop = mapPitchToCanvas(0, -GOAL_WIDTH / 2, w, h)
    const goalBottom = mapPitchToCanvas(0, GOAL_WIDTH / 2, w, h)
    ctx.beginPath()
    ctx.moveTo(goalTop.px, goalTop.py)
    ctx.lineTo(goalBottom.px, goalBottom.py)
    ctx.stroke()

    // Penalty area (16.5m) and box (5.5m)
    const penaltyDepth = 16.5
    const sixYard = 5.5

    const topPA = mapPitchToCanvas(0, -PITCH_WIDTH / 2 + 16.5, w, h)
    const bottomPA = mapPitchToCanvas(0, PITCH_WIDTH / 2 - 16.5, w, h)
    const paRight = mapPitchToCanvas(penaltyDepth, 0, w, h)

    ctx.beginPath()
    ctx.moveTo(topPA.px, topPA.py)
    ctx.lineTo(paRight.px, topPA.py)
    ctx.lineTo(paRight.px, bottomPA.py)
    ctx.lineTo(bottomPA.px, bottomPA.py)
    ctx.stroke()

    const top6 = mapPitchToCanvas(0, -sixYard, w, h)
    const bottom6 = mapPitchToCanvas(0, sixYard, w, h)
    const sixRight = mapPitchToCanvas(5.5, 0, w, h)

    ctx.beginPath()
    ctx.moveTo(top6.px, top6.py)
    ctx.lineTo(sixRight.px, top6.py)
    ctx.lineTo(sixRight.px, bottom6.py)
    ctx.lineTo(bottom6.px, bottom6.py)
    ctx.stroke()

    // Penalty spot (11m)
    const penSpot = mapPitchToCanvas(11, 0, w, h)
    ctx.beginPath()
    ctx.arc(penSpot.px, penSpot.py, 3, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(255,255,255,0.9)'
    ctx.fill()
  }, [playersMemo])

  const handlePitchMouseMove: React.MouseEventHandler<HTMLDivElement> = (e) => {
    const pitchEl = pitchRef.current
    if (!pitchEl) return

    const rect = pitchEl.getBoundingClientRect()
    const px = e.clientX - rect.left
    const py = e.clientY - rect.top

    if (px < 0 || py < 0 || px > rect.width || py > rect.height) {
      setHoverXG(null)
      setHoverPos(null)
      return
    }

    const { x, y } = mapCanvasToPitch(px, py, rect.width, rect.height)

    if (x < 0 || x > PITCH_LENGTH || Math.abs(y) > PITCH_WIDTH / 2) {
      setHoverXG(null)
      setHoverPos(null)
      return
    }

    const xg = computeXG(x, y, players)
    setHoverXG(xg)
    setHoverPos({ x, y })

    if (draggingId !== null) {
      setPlayers((prev) =>
        prev.map((p) =>
          p.id === draggingId
            ? {
                ...p,
                x: Math.min(Math.max(x, 0), PITCH_LENGTH),
                y: Math.min(Math.max(y, -PITCH_WIDTH / 2), PITCH_WIDTH / 2),
              }
            : p,
        ),
      )
    }
  }

  const handlePitchMouseLeave: React.MouseEventHandler<HTMLDivElement> = () => {
    setHoverXG(null)
    setHoverPos(null)
    setDraggingId(null)
  }

  const handleMouseUp: React.MouseEventHandler<HTMLDivElement> = () => {
    setDraggingId(null)
  }

  return (
    <div className="app-root">
      <header className="app-header">
        <h1>Expected Goals (xG) and Defensive Shape</h1>
        <p>
          Drag the defending team (including goalkeeper) to see how their shape
          changes the shot xG.
        </p>
      </header>

      <main className="layout">
        <section className="pitch-panel">
          <div
            className="pitch-container"
            ref={pitchRef}
            onMouseMove={handlePitchMouseMove}
            onMouseLeave={handlePitchMouseLeave}
            onMouseUp={handleMouseUp}
          >
            <canvas
              ref={canvasRef}
              className="pitch-canvas"
            />

            {players.map((p) => {
              const rect = pitchRef.current?.getBoundingClientRect()
              const width = rect?.width ?? 1
              const height = rect?.height ?? 1
              const { px, py } = mapPitchToCanvas(p.x, p.y, width, height)

              return (
                <button
                  key={p.id}
                  type="button"
                  className={`player-marker ${
                    p.role === 'GK' ? 'player-gk' : 'player-def'
                  }`}
                  style={{
                    left: px,
                    top: py,
                  }}
                  onMouseDown={() => setDraggingId(p.id)}
                >
                  {p.role === 'GK' ? 'GK' : p.id}
                </button>
              )
            })}
          </div>
        </section>

        <aside className="side-panel">
          <div className="card">
            <h2>Shot xG at cursor</h2>
            {hoverXG != null && hoverPos ? (
              <>
                <div className="xg-value">
                  {(hoverXG * 100).toFixed(1)}%
                  <span className="xg-label">chance of goal</span>
                </div>
                <div className="coords">
                  <span>
                    x: {hoverPos.x.toFixed(1)} m &nbsp; y:{' '}
                    {hoverPos.y.toFixed(1)} m
                  </span>
                </div>
              </>
            ) : (
              <p className="muted">
                Move your mouse over the pitch to inspect shot xG.
              </p>
            )}
          </div>

          <div className="card">
            <h2>How it works</h2>
            <ul className="info-list">
              <li>
                <strong>Location</strong> – base xG from shot distance and
                angle to the posts.
              </li>
              <li>
                <strong>Open goal</strong> – fraction of the goal mouth not
                blocked by defenders or the goalkeeper.
              </li>
              <li>
                <strong>Pressure</strong> – number of defenders between the ball
                and goal, and distance to the closest defender.
              </li>
              <li>
                <strong>GK coverage</strong> – how well the keeper&apos;s
                position covers the best open part of the goal.
              </li>
            </ul>
          </div>

          <div className="card">
            <h2>Tips</h2>
            <ul className="info-list">
              <li>
                <strong>Drag players</strong> to explore different defensive
                shapes.
              </li>
              <li>
                <strong>Watch the colours</strong> – deep red zones are very
                high xG, blues are low.
              </li>
            </ul>
          </div>
        </aside>
      </main>
    </div>
  )
}

export default App
