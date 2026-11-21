const SIDES = { left: [11, 13, 15], right: [12, 14, 16] }
const BONES = [[11, 12], [11, 23], [12, 24], [23, 24]]
const TRAIL = 45
const GLOW_AT = 0.08

let start
let trails
let peak

export function resetArms() {
  start = undefined
  trails = { left: [], right: [] }
  peak = { left: 0, right: 0 }
  hideArms()
}

export function hideArms() {
  document.querySelector("#viz-arms")?.remove()
}

function bars() {
  let box = document.querySelector("#viz-arms")
  if (!box) {
    box = document.createElement("div")
    box.id = "viz-arms"
    box.innerHTML = ["left", "right"].map((side) => `<div class="viz-arm" data-side="${side}"><span>${side === "left" ? "Left" : "Right"} arm drop</span><div class="viz-arm-bar"><div class="viz-arm-fill"></div></div><strong></strong></div>`).join("")
    document.querySelector("#preview").append(box)
  }
  return box
}

const seen = (p) => p && (p.visibility ?? 1) >= 0.5

export function drawArms(canvas, context, video, pose, holding) {
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  const w = canvas.width
  const h = canvas.height
  const at = (p) => [p.x * w, p.y * h]
  if (!pose.length) return
  const shoulderWidth = seen(pose[11]) && seen(pose[12]) ? Math.hypot((pose[11].x - pose[12].x) * w, (pose[11].y - pose[12].y) * h) : 0
  const drop = { left: 0, right: 0 }
  if (holding && shoulderWidth > 10) {
    if (!start && seen(pose[15]) && seen(pose[16])) start = { left: pose[15].y * h, right: pose[16].y * h, width: shoulderWidth }
    if (start) {
      for (const side of ["left", "right"]) {
        const wrist = pose[SIDES[side][2]]
        if (!seen(wrist)) continue
        drop[side] = Math.max(0, (wrist.y * h - start[side]) / start.width)
        peak[side] = Math.max(peak[side], drop[side])
        trails[side].push(at(wrist))
        if (trails[side].length > TRAIL) trails[side].shift()
      }
    }
  }
  const weaker = drop.left - drop.right > GLOW_AT ? "left" : drop.right - drop.left > GLOW_AT ? "right" : null

  context.lineCap = "round"
  context.strokeStyle = "#63f0d988"
  context.lineWidth = 3
  for (const [a, b] of BONES) {
    if (!seen(pose[a]) || !seen(pose[b])) continue
    context.beginPath()
    context.moveTo(...at(pose[a]))
    context.lineTo(...at(pose[b]))
    context.stroke()
  }
  if (start) {
    context.setLineDash([10, 8])
    context.lineWidth = 2
    context.strokeStyle = "#ffffffcc"
    for (const side of ["left", "right"]) {
      const wrist = pose[SIDES[side][2]]
      const x = seen(wrist) ? wrist.x * w : w / 2
      context.beginPath()
      context.moveTo(x - 90, start[side])
      context.lineTo(x + 90, start[side])
      context.stroke()
    }
    context.setLineDash([])
  }
  for (const side of ["left", "right"]) {
    const color = side === weaker ? "#ff4b3a" : "#63f0d9"
    const trail = trails[side] ?? []
    trail.forEach(([x, y], i) => {
      context.globalAlpha = (i + 1) / trail.length
      context.fillStyle = color
      context.beginPath()
      context.arc(x, y, 3 + 4 * ((i + 1) / trail.length), 0, Math.PI * 2)
      context.fill()
    })
    context.globalAlpha = 1
    const joints = SIDES[side].map((i) => pose[i])
    if (joints.some((p) => !seen(p))) continue
    context.shadowColor = side === weaker ? "#ff4b3a" : "transparent"
    context.shadowBlur = side === weaker ? 24 : 0
    context.strokeStyle = color
    context.fillStyle = color
    context.lineWidth = side === weaker ? 7 : 5
    context.beginPath()
    context.moveTo(...at(joints[0]))
    for (const p of joints.slice(1)) context.lineTo(...at(p))
    context.stroke()
    for (const p of joints) {
      context.beginPath()
      context.arc(...at(p), 6, 0, Math.PI * 2)
      context.fill()
    }
    context.shadowBlur = 0
  }
  if (!holding) return
  const box = bars()
  for (const side of ["left", "right"]) {
    const row = box.querySelector(`[data-side="${side}"]`)
    row.classList.toggle("weaker", side === weaker)
    row.querySelector(".viz-arm-fill").style.width = `${Math.min(100, Math.round(drop[side] / 0.5 * 100))}%`
    row.querySelector("strong").textContent = drop[side].toFixed(2)
  }
}

export function chart(canvas, trace) {
  if (!trace?.t?.length) return false
  const ratio = window.devicePixelRatio || 1
  const width = canvas.clientWidth || 520
  const height = canvas.clientHeight || 220
  canvas.width = width * ratio
  canvas.height = height * ratio
  const c = canvas.getContext("2d")
  c.scale(ratio, ratio)
  const pad = { left: 46, right: 12, top: 14, bottom: 28 }
  const values = [...trace.left, ...trace.right, 0]
  const low = Math.min(...values, -0.05)
  const high = Math.max(...values, 0.3)
  const tMax = Math.max(...trace.t, 1)
  const x = (t) => pad.left + (t / tMax) * (width - pad.left - pad.right)
  const y = (v) => pad.top + ((v - low) / (high - low)) * (height - pad.top - pad.bottom)
  c.font = "12px system-ui, sans-serif"
  c.fillStyle = "#4c655d"
  c.strokeStyle = "#acbcb0"
  c.lineWidth = 1
  c.beginPath()
  c.moveTo(pad.left, y(0))
  c.lineTo(width - pad.right, y(0))
  c.stroke()
  c.fillText("start", 6, y(0) + 4)
  c.fillText(`${high.toFixed(2)} ↓`, 4, y(high) - 2)
  c.fillText("0 s", pad.left, height - 8)
  c.fillText(`${tMax.toFixed(0)} s`, width - pad.right - 22, height - 8)
  for (const [side, color] of [["left", "#196552"], ["right", "#c26a12"]]) {
    c.strokeStyle = color
    c.lineWidth = 2.5
    c.beginPath()
    trace[side].forEach((v, i) => (i ? c.lineTo(x(trace.t[i]), y(v)) : c.moveTo(x(trace.t[i]), y(v))))
    c.stroke()
  }
  return true
}
