import * as THREE from "three"
import { OrbitControls } from "three/addons/controls/OrbitControls.js"

const EXAGGERATE = 2
let current

export function stopFace3d() {
  if (!current) return
  cancelAnimationFrame(current.frame)
  current.resizeObserver.disconnect()
  current.renderer.dispose()
  current.renderer.domElement.remove()
  current = undefined
}

function heatColor(value) {
  return new THREE.Color().setHSL((120 * (1 - Math.min(1, Math.max(0, value)))) / 360, 0.9, 0.52)
}

export function showFace3d(container, overlay, tessellation, aspect, playButton) {
  stopFace3d()
  const neutral = overlay.neutral
  const smile = overlay.smile
  if (!neutral?.length || !smile?.length || !tessellation.length) return false
  const cx = neutral.reduce((s, p) => s + p[0], 0) / neutral.length
  const cy = neutral.reduce((s, p) => s + p[1], 0) / neutral.length
  const cz = neutral.reduce((s, p) => s + p[2], 0) / neutral.length
  const scale = 10
  const place = (p) => [(p[0] - cx) * aspect * scale, -(p[1] - cy) * scale, -(p[2] - cz) * aspect * scale]
  const from = neutral.map(place)
  const placedSmile = smile.map(place)
  const shift = placedSmile[168].map((v, k) => v - from[168][k])
  const to = placedSmile.map((p, i) => p.map((v, k) => from[i][k] + (v - shift[k] - from[i][k]) * EXAGGERATE))
  const edges = tessellation.filter(([a, b]) => a < from.length && b < from.length)

  const positions = new Float32Array(edges.length * 6)
  const colors = new Float32Array(edges.length * 6)
  edges.forEach(([a, b], e) => {
    for (const [k, index] of [[0, a], [1, b]]) {
      const color = heatColor(overlay.asymmetry?.[index] ?? 0)
      colors.set([color.r, color.g, color.b], e * 6 + k * 3)
    }
  })
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3))
  const mesh = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95 }))

  const scene = new THREE.Scene()
  scene.add(mesh)
  const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100)
  camera.position.set(0, 0, 9)
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1))
  container.append(renderer.domElement)
  const controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.enablePan = false
  controls.autoRotate = true
  controls.autoRotateSpeed = 1.2
  controls.minDistance = 4
  controls.maxDistance = 16

  function resize() {
    const width = container.clientWidth || 360
    const height = container.clientHeight || 320
    renderer.setSize(width, height)
    camera.aspect = width / height
    camera.updateProjectionMatrix()
  }
  const resizeObserver = new ResizeObserver(resize)
  resizeObserver.observe(container)
  resize()

  let playing = true
  let phase = 0
  let last = performance.now()
  function setShape(k) {
    edges.forEach(([a, b], e) => {
      for (const [slot, index] of [[0, a], [1, b]]) {
        for (let d = 0; d < 3; d++) positions[e * 6 + slot * 3 + d] = from[index][d] + (to[index][d] - from[index][d]) * k
      }
    })
    geometry.attributes.position.needsUpdate = true
  }
  setShape(1)
  if (playButton) {
    playButton.textContent = "Pause smile"
    playButton.onclick = () => {
      playing = !playing
      playButton.textContent = playing ? "Pause smile" : "Play smile"
    }
  }
  current = { renderer, resizeObserver, frame: 0 }
  function animate(now) {
    if (playing) {
      phase += (now - last) / 1000
      setShape((1 - Math.cos(phase * Math.PI * 0.8)) / 2)
    }
    last = now
    controls.update()
    renderer.render(scene, camera)
    current.frame = requestAnimationFrame(animate)
  }
  current.frame = requestAnimationFrame(animate)
  return true
}
