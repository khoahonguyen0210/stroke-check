const MIDLINE_TOP = 10
const MIDLINE_CHIN = 152
const NOSE_BRIDGE = 168
const RIGHT_EYE = 33
const LEFT_EYE = 263
const LEVELS = 8

let tessellation = []
let neutralSum
let neutralCount = 0
let neutral
let pairs
let eyeWidth = 1
let smoothed = 0
let aspect = 4 / 3

export function setTessellation(edges) {
  tessellation = (edges ?? []).map((edge) => [edge.start, edge.end])
}

export function getTessellation() {
  return tessellation
}

export function videoAspect() {
  return aspect
}

export function heat(value) {
  return `hsl(${Math.round(120 * (1 - Math.min(1, Math.max(0, value))))}, 90%, 52%)`
}

function gauge() {
  let box = document.querySelector("#viz-gauge")
  if (!box) {
    box = document.createElement("div")
    box.id = "viz-gauge"
    box.innerHTML = '<span class="viz-gauge-title">Face asymmetry</span><div class="viz-gauge-bar"><div class="viz-gauge-fill"></div></div><strong class="viz-gauge-value"></strong><span class="viz-gauge-side"></span>'
    document.querySelector("#preview").append(box)
  }
  return box
}

export function resetFace() {
  neutralSum = undefined
  neutralCount = 0
  neutral = undefined
  pairs = undefined
  smoothed = 0
  hideFace()
}

export function hideFace() {
  document.querySelector("#viz-gauge")?.remove()
}

function buildPairs(points) {
  const top = points[MIDLINE_TOP]
  const chin = points[MIDLINE_CHIN]
  const length = Math.hypot(chin.x - top.x, chin.y - top.y) || 1
  const normal = { x: -(chin.y - top.y) / length, y: (chin.x - top.x) / length }
  const offset = points.map((p) => (p.x - top.x) * normal.x + (p.y - top.y) * normal.y)
  const leftSign = Math.sign(offset[LEFT_EYE]) || 1
  const left = []
  const right = []
  offset.forEach((value, index) => {
    if (value * leftSign > 0.002) left.push(index)
    else if (value * leftSign < -0.002) right.push(index)
  })
  return left.map((i) => {
    const rx = points[i].x - 2 * offset[i] * normal.x
    const ry = points[i].y - 2 * offset[i] * normal.y
    let best = right[0]
    let bestDistance = Infinity
    for (const j of right) {
      const d = (points[j].x - rx) ** 2 + (points[j].y - ry) ** 2
      if (d < bestDistance) {
        bestDistance = d
        best = j
      }
    }
    return [i, best]
  })
}

function measure(points) {
  const moved = points.map((p, i) => {
    const dx = (p.x - points[NOSE_BRIDGE].x) - (neutral[i].x - neutral[NOSE_BRIDGE].x)
    const dy = (p.y - points[NOSE_BRIDGE].y) - (neutral[i].y - neutral[NOSE_BRIDGE].y)
    return Math.hypot(dx, dy) / eyeWidth
  })
  const asymmetry = new Float32Array(points.length)
  let difference = 0
  let movement = 0
  let leftTotal = 0
  let rightTotal = 0
  for (const [i, j] of pairs) {
    const top = Math.max(moved[i], moved[j], 0.02)
    const value = Math.abs(moved[i] - moved[j]) / top
    asymmetry[i] = Math.max(asymmetry[i], value)
    asymmetry[j] = Math.max(asymmetry[j], value)
    difference += Math.abs(moved[i] - moved[j])
    movement += top
    leftTotal += moved[i]
    rightTotal += moved[j]
  }
  return { asymmetry, overall: movement ? difference / movement : 0, leftTotal, rightTotal }
}

function drawMesh(context, width, height, points, colorOf) {
  const buckets = Array.from({ length: LEVELS }, () => [])
  for (const [a, b] of tessellation) {
    if (!points[a] || !points[b]) continue
    buckets[Math.min(LEVELS - 1, Math.floor(colorOf(a, b) * LEVELS))].push([a, b])
  }
  context.lineWidth = 1
  buckets.forEach((edges, level) => {
    if (!edges.length) return
    context.strokeStyle = heat((level + 0.5) / LEVELS)
    context.globalAlpha = 0.55 + 0.45 * (level / LEVELS)
    context.beginPath()
    for (const [a, b] of edges) {
      context.moveTo(points[a].x * width, points[a].y * height)
      context.lineTo(points[b].x * width, points[b].y * height)
    }
    context.stroke()
  })
  context.globalAlpha = 1
}

export function drawFace(canvas, context, video, points, phase) {
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  if (video.videoHeight) aspect = video.videoWidth / video.videoHeight
  if (!points.length) return
  const box = gauge()
  if (phase === "relax" || !neutralCount) {
    if (phase === "relax") {
      neutralSum ??= points.map(() => ({ x: 0, y: 0 }))
      points.forEach((p, i) => {
        neutralSum[i].x += p.x
        neutralSum[i].y += p.y
      })
      neutralCount += 1
    }
    drawMesh(context, canvas.width, canvas.height, points, () => 0)
    box.querySelector(".viz-gauge-value").textContent = "…"
    box.querySelector(".viz-gauge-side").textContent = "Learning your neutral face"
    box.querySelector(".viz-gauge-fill").style.height = "0%"
    return
  }
  if (!pairs) {
    neutral = neutralSum.map((p) => ({ x: p.x / neutralCount, y: p.y / neutralCount }))
    eyeWidth = Math.hypot(neutral[LEFT_EYE].x - neutral[RIGHT_EYE].x, neutral[LEFT_EYE].y - neutral[RIGHT_EYE].y) || 1
    pairs = buildPairs(neutral)
  }
  const { asymmetry, overall, leftTotal, rightTotal } = measure(points)
  drawMesh(context, canvas.width, canvas.height, points, (a, b) => (asymmetry[a] + asymmetry[b]) / 2)
  smoothed = smoothed * 0.8 + overall * 0.2
  const fill = box.querySelector(".viz-gauge-fill")
  fill.style.height = `${Math.round(Math.min(1, smoothed) * 100)}%`
  fill.style.background = heat(smoothed)
  box.querySelector(".viz-gauge-value").textContent = `${Math.round(smoothed * 100)}%`
  const weaker = leftTotal < rightTotal ? "Left" : "Right"
  box.querySelector(".viz-gauge-side").textContent = smoothed >= 0.2 ? `${weaker} side moving less` : "Both sides moving evenly"
}
