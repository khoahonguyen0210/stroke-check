const version = "0.10.22"
const video = document.querySelector("video")
const canvas = document.querySelector("canvas")
const context = canvas.getContext("2d")
const status = document.querySelector("#status")

async function loadVision() {
  const root = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${version}`
  const vision = await import(`${root}/vision_bundle.mjs`)
  return { vision, files: await vision.FilesetResolver.forVisionTasks(`${root}/wasm`) }
}

async function createFace({ vision, files }) {
  return vision.FaceLandmarker.createFromOptions(files, {
    baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" },
    runningMode: "VIDEO",
  })
}

function drawFace(points) {
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  context.fillStyle = "#40ff90"
  for (const point of points) {
    context.fillRect(point.x * canvas.width, point.y * canvas.height, 2, 2)
  }
}

document.querySelector("#start").onclick = async (event) => {
  event.target.disabled = true
  try {
    const face = await createFace(await loadVision())
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true })
    await video.play()
    document.querySelector("#preview").hidden = false
    function frame(now) {
      drawFace(face.detectForVideo(video, now).faceLandmarks[0] ?? [])
      requestAnimationFrame(frame)
    }
    requestAnimationFrame(frame)
  } catch (error) {
    status.textContent = error.message
    event.target.disabled = false
  }
}
