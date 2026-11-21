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

const $ = (selector) => document.querySelector(selector)
const sentence = "The quick brown fox jumps over the lazy dog near the river bank."
let visionTasks

async function createTasks() {
  const { vision, files } = await loadVision()
  const root = "https://storage.googleapis.com/mediapipe-models"
  const options = (path) => ({ runningMode: "VIDEO", baseOptions: { modelAssetPath: root + path } })
  return {
    face: await createFace({ vision, files }),
    pose: await vision.PoseLandmarker.createFromOptions(files, options("/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")),
    hands: await vision.HandLandmarker.createFromOptions(files, {
      ...options("/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"), numHands: 2,
    }),
  }
}

const xyz = ({ x, y, z }) => [x, y, z]

function capture(duration, message, read) {
  const frames = []
  const start = performance.now()
  let lastVideoTime = -1
  return new Promise((resolve, reject) => {
    function frame(now) {
      const elapsed = now - start
      if (elapsed >= duration) {
        resolve(frames)
        return
      }
      try {
        $("#prompt").textContent = `${message} (${Math.ceil((duration - elapsed) / 1000)} s)`
        if (lastVideoTime !== video.currentTime) {
          lastVideoTime = video.currentTime
          frames.push({ t: Math.max(0, elapsed / 1000), ...read(now) })
        }
        requestAnimationFrame(frame)
      } catch (error) {
        reject(error)
      }
    }
    requestAnimationFrame(frame)
  })
}

function speech(stream) {
  return new Promise((resolve, reject) => {
    const recorder = new MediaRecorder(new MediaStream(stream.getAudioTracks()), { mimeType: "audio/webm" })
    const chunks = []
    const done = $("#done")
    let timer
    const stop = () => {
      if (recorder.state === "recording") recorder.stop()
    }
    const cleanup = () => {
      clearInterval(timer)
      done.hidden = true
      done.onclick = null
    }
    recorder.ondataavailable = ({ data }) => chunks.push(data)
    recorder.onerror = ({ error }) => { cleanup(); reject(error) }
    recorder.onstop = () => { cleanup(); resolve(new Blob(chunks, { type: recorder.mimeType })) }
    recorder.start()
    done.hidden = false
    done.onclick = stop
    const started = performance.now()
    $("#prompt").textContent = sentence + " (8 s)"
    timer = setInterval(() => {
      const remaining = Math.ceil((8000 - (performance.now() - started)) / 1000)
      if (remaining <= 0) stop()
      else $("#prompt").textContent = `${sentence} (${remaining} s)`
    }, 100)
  })
}

async function postRecording(stream, tasks, collecting) {
  const face = []
  const phases = [["relax", "Relax your face"], ["smile", "Smile big"], ["brows", "Raise your eyebrows"]]
  for (const [index, [phase, message]] of phases.entries()) {
    face.push(...await capture(3000, message, (now) => {
      const points = tasks.face.detectForVideo(video, now).faceLandmarks[0] ?? []
      drawFace(points)
      return { phase, landmarks: points.map(xyz) }
    }).then((frames) => frames.map((frame) => ({ ...frame, t: frame.t + index * 3 }))))
  }
  context.clearRect(0, 0, canvas.width, canvas.height)
  const arms = await capture(10000, "Raise both arms straight out, palms up, and hold still", (now) => {
    const pose = tasks.pose.detectForVideo(video, now).landmarks[0] ?? []
    const hands = tasks.hands.detectForVideo(video, now)
    return {
      pose: pose.map((point) => [...xyz(point), point.visibility]),
      hands: hands.landmarks.map((points, index) => ({
        side: hands.handedness[index][0].categoryName, landmarks: points.map(xyz),
      })),
    }
  })
  const audio = await speech(stream)
  stream.getTracks().forEach((track) => track.stop())
  $("#preview").hidden = true
  $("#prompt").textContent = "Submitting recording…"
  const data = new FormData()
  data.set("meta", JSON.stringify({
    name: $("#name").value.trim(), user_agent: navigator.userAgent,
    glasses: $("#glasses").checked, lighting: $("#lighting").value,
  }))
  data.set("face", JSON.stringify({ fps: face.length / 9, frames: face }))
  data.set("arms", JSON.stringify({ fps: arms.length / 10, frames: arms }))
  data.set("audio", audio, "speech.webm")
  if (collecting) data.set("label", $("#label").value)
  const response = await fetch(collecting ? "/api/collect" : "/api/check", { method: "POST", body: data })
  if (!response.ok) throw new Error("The recording could not be submitted.")
  const result = await response.json()
  if (collecting) status.textContent = `Saved to ${result.path}`
  else renderResult(result)
}

function renderResult(result) {
  status.textContent = result.retake_reason ?? result.overall
}

$("#collect").onchange = (event) => { $("#labels").hidden = !event.target.checked }
$("#setup").onsubmit = async (event) => {
  event.preventDefault()
  let stream
  const collecting = $("#collect").checked
  const inputs = [...$("#setup").elements]
  inputs.forEach((input) => { input.disabled = true })
  status.textContent = "Loading camera and models…"
  try {
    visionTasks ??= createTasks().catch((error) => { visionTasks = null; throw error })
    const tasks = await visionTasks
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true })
    video.srcObject = stream
    await video.play()
    $("#preview").hidden = false
    status.textContent = ""
    await postRecording(stream, tasks, collecting)
  } catch (error) {
    status.textContent = error.message
  } finally {
    stream?.getTracks().forEach((track) => track.stop())
    video.srcObject = null
    $("#preview").hidden = true
    $("#prompt").textContent = ""
    inputs.forEach((input) => { input.disabled = false })
  }
}
