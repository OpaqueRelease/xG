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

type XGComponents = {
  baseLoc: number
  F_open: number
  F_def: number
  F_gk: number
}

type PitchSize = {
  width: number
  height: number
}

// Pitch & goal dimensions (in meters)
const PITCH_LENGTH = 52.5 // half pitch, standard 105m total
const PITCH_WIDTH = 68
const GOAL_WIDTH = 7.32

// Model tuning constants (hand-tuned calibration)
// K_DISTANCE controls base xG vs distance. Larger values increase close-range xG.
const K_DISTANCE = 32
const BETA_OPEN = 1.2
// Even when the goal rays look fully blocked, there is still a tiny chance
// to score by shooting over/around defenders.
const MIN_OPEN_FACTOR = 0.05
// Global scale applied to all shot types (open play, crosses, headers).
// Keep at 1 for now; overall xG saturation is handled by the final logistic.
const GLOBAL_XG_SCALE = 1
const A_DEF_BETWEEN = 0.4
const B_PRESSURE = 0.8
const MIN_HEADER_FACTOR = 0.05 // residual headed chance even when far / very hard

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

  const baseCentral = K_DISTANCE / (safeD * safeD)

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

// Core xG components shared by all shot types (open play, cross, header, etc.)
function computeXGComponents(
  shotX: number,
  shotY: number,
  players: Player[],
): XGComponents | null {
  if (shotX <= 0) return null

  const d = distance(shotX, shotY, 0, 0)
  const alpha = computeAngleToPosts(shotX, shotY)

  if (!Number.isFinite(d) || d <= 0) return null

  const baseLoc = computeBaseLocationTerm(d, alpha)

  const openGoal = computeOpenGoal(shotX, shotY, players)
  const fOpen = openGoal.fractionOpen

  // Even if fractionOpen is computed as 0, shots can still go over/around defenders.
  // We therefore never reduce xG all the way to 0 purely from blocking.
  const F_openBase = fOpen <= 0 ? 0 : Math.pow(fOpen, BETA_OPEN)
  const F_open = F_openBase * (1 - MIN_OPEN_FACTOR) + MIN_OPEN_FACTOR

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

  return { baseLoc, F_open, F_def, F_gk }
}

// Main model entry point: choose between open-play, cross, and cross-header variants
function computeXG(
  shotX: number,
  shotY: number,
  players: Player[],
  opts: { isCross: boolean; isHeader: boolean },
): number {
  const core = computeXGComponents(shotX, shotY, players)
  if (!core) return 0

  let { baseLoc, F_open, F_def, F_gk } = core

  // Open-play (default) model uses the core components directly.
  // Cross / header variants re-weight the base location term to reflect
  // mechanical difficulty of the shot type.
  const d = distance(shotX, shotY, 0, 0)

  // Difficulty from finishing a cross (with foot), relative to a fully controlled shot.
  let crossFactor = 1
  if (opts.isCross) {
    if (d < 8) {
      crossFactor = 0.85
    } else if (d < 16.5) {
      crossFactor = 0.8
    } else if (d < 25) {
      crossFactor = 0.7
    } else {
      crossFactor = 0.6
    }
  }

  // Additional difficulty for headers: close-range headers are still reasonably dangerous,
  // but difficulty rises quickly with distance and approaches ~0 around 16m.
  let headerFactor = 1
  if (opts.isHeader) {
    if (d <= 8) {
      headerFactor = 0.7
    } else if (d <= 16) {
      const t = (d - 8) / 8 // 0 at 8m, 1 at 16m
      const falloff = 0.7 * (1 - t * t)
      headerFactor = Math.max(MIN_HEADER_FACTOR, falloff)
    } else {
      headerFactor = MIN_HEADER_FACTOR
    }
  }

  baseLoc *= crossFactor * headerFactor

  let raw = baseLoc * F_open * F_def * F_gk * GLOBAL_XG_SCALE
  if (!Number.isFinite(raw) || raw <= 0) return 0

  // Convert raw scoring "intensity" into probability with a smooth saturation.
  // This avoids an arbitrary hard cap (like 0.95) while keeping values < 1.
  const xg = raw / (1 + raw)
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
  // Rotate pitch 90 degrees counter-clockwise for display:
  // - Horizontal screen axis corresponds to pitch width (y)
  // - Vertical screen axis corresponds to pitch length (x), with goal at the bottom
  const px = width / 2 + (y / PITCH_WIDTH) * width
  const py = height - (x / PITCH_LENGTH) * height
  return { px, py }
}

function mapCanvasToPitch(
  px: number,
  py: number,
  width: number,
  height: number,
): { x: number; y: number } {
  const y = ((px - width / 2) / width) * PITCH_WIDTH
  const x = ((height - py) / height) * PITCH_LENGTH
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

  const [isCross, setIsCross] = useState(false)
  const [isHeader, setIsHeader] = useState(false)

  const [draggingId, setDraggingId] = useState<number | null>(null)
  const pitchRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [pitchSize, setPitchSize] = useState<PitchSize>({ width: 1, height: 1 })

  const playersMemo = useMemo(() => players, [players])

  // Measure pitch container size on mount and when window resizes
  useEffect(() => {
    const updateSize = () => {
      const el = pitchRef.current
      if (!el) return
      const rect = el.getBoundingClientRect()
      if (rect.width > 0 && rect.height > 0) {
        setPitchSize({ width: rect.width, height: rect.height })
      }
    }

    updateSize()
    window.addEventListener('resize', updateSize)
    return () => {
      window.removeEventListener('resize', updateSize)
    }
  }, [])

  // Draw heatmap and pitch markings
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    // Keep the internal canvas resolution in sync with the measured pitch size
    const displayWidth = Math.max(1, Math.floor(pitchSize.width))
    const displayHeight = Math.max(1, Math.floor(pitchSize.height))
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

    // xG heatmap grid (very high resolution, adapts to canvas size)
    const targetCellSize = 6 // pixels
    const cellsX = Math.max(60, Math.floor(w / targetCellSize))
    const cellsY = Math.max(36, Math.floor(h / targetCellSize))
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

        const xg = computeXG(x, y, playersMemo, {
          isCross,
          isHeader,
        })
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

    // Goal frame (simple 2D rectangle extending 1m into the pitch)
    const goalDepth = 1
    const goalTopIn = mapPitchToCanvas(goalDepth, -GOAL_WIDTH / 2, w, h)
    const goalBottomIn = mapPitchToCanvas(goalDepth, GOAL_WIDTH / 2, w, h)

    ctx.save()
    ctx.strokeStyle = 'rgba(255,255,255,0.95)'
    ctx.lineWidth = 3
    ctx.beginPath()
    // Left post
    ctx.moveTo(goalTop.px, goalTop.py)
    ctx.lineTo(goalTopIn.px, goalTopIn.py)
    // Crossbar
    ctx.lineTo(goalBottomIn.px, goalBottomIn.py)
    // Right post
    ctx.lineTo(goalBottom.px, goalBottom.py)
    ctx.stroke()
    ctx.restore()

    // Penalty area (16.5m deep, 40.32m wide) and goal area (5.5m deep, 18.32m wide)
    const penaltyDepth = 16.5
    const penaltyHalfWidth = 40.32 / 2
    const sixYardDepth = 5.5
    const sixYardHalfWidth = 18.32 / 2

    // Penalty area rectangle (x from 0 to 16.5, y from -20.16 to +20.16)
    const paTopGoal = mapPitchToCanvas(0, -penaltyHalfWidth, w, h)
    const paBottomGoal = mapPitchToCanvas(0, penaltyHalfWidth, w, h)
    const paTopDepth = mapPitchToCanvas(penaltyDepth, -penaltyHalfWidth, w, h)
    const paBottomDepth = mapPitchToCanvas(penaltyDepth, penaltyHalfWidth, w, h)

    ctx.beginPath()
    ctx.moveTo(paTopGoal.px, paTopGoal.py)
    ctx.lineTo(paTopDepth.px, paTopDepth.py)
    ctx.lineTo(paBottomDepth.px, paBottomDepth.py)
    ctx.lineTo(paBottomGoal.px, paBottomGoal.py)
    ctx.closePath()
    ctx.stroke()

    // Six-yard box rectangle (x from 0 to 5.5, y from -9.16 to +9.16)
    const sixTopGoal = mapPitchToCanvas(0, -sixYardHalfWidth, w, h)
    const sixBottomGoal = mapPitchToCanvas(0, sixYardHalfWidth, w, h)
    const sixTopDepth = mapPitchToCanvas(sixYardDepth, -sixYardHalfWidth, w, h)
    const sixBottomDepth = mapPitchToCanvas(sixYardDepth, sixYardHalfWidth, w, h)

    ctx.beginPath()
    ctx.moveTo(sixTopGoal.px, sixTopGoal.py)
    ctx.lineTo(sixTopDepth.px, sixTopDepth.py)
    ctx.lineTo(sixBottomDepth.px, sixBottomDepth.py)
    ctx.lineTo(sixBottomGoal.px, sixBottomGoal.py)
    ctx.closePath()
    ctx.stroke()

    // Penalty spot (11m)
    const penSpot = mapPitchToCanvas(11, 0, w, h)
    ctx.beginPath()
    ctx.arc(penSpot.px, penSpot.py, 3, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(255,255,255,0.9)'
    ctx.fill()
  }, [playersMemo, isCross, isHeader, pitchSize])

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

    const xg = computeXG(x, y, players, {
      isCross,
      isHeader,
    })
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
              const { px, py } = mapPitchToCanvas(
                p.x,
                p.y,
                pitchSize.width,
                pitchSize.height,
              )

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
                <div className="shot-type">
                  <span className="shot-type-label">Shot context:</span>{' '}
                  {isCross ? (isHeader ? 'Cross → Header' : 'Cross → Shot') : 'Open play'}
                </div>
              </>
            ) : (
              <p className="muted">
                Move your mouse over the pitch to inspect shot xG.
              </p>
            )}
            <div className="shot-toggles">
              <label>
                <input
                  type="checkbox"
                  checked={isCross}
                  onChange={(e) => {
                    const next = e.target.checked
                    setIsCross(next)
                    if (!next) {
                      setIsHeader(false)
                    }
                  }}
                />{' '}
                Cross
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={isHeader}
                  disabled={!isCross}
                  onChange={(e) => setIsHeader(e.target.checked)}
                />{' '}
                Header
              </label>
            </div>
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
