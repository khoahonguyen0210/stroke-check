const version = "0.10.21"
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
let visionTasks

async function createTasks() {
  const { vision, files } = await loadVision()
  const root = "https://storage.googleapis.com/mediapipe-models"
  const options = (path) => ({ runningMode: "VIDEO", baseOptions: { modelAssetPath: root + path } })
  return {
    face: await createFace({ vision, files }),
    pose: await vision.PoseLandmarker.createFromOptions(files, options("/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")),
  }
}

const xyz = ({ x, y, z }) => [x, y, z]

function capture(duration, message, read) {
  const frames = []
  const start = performance.now()
  $("#prompt").textContent = message
  $("#capture-progress").max = duration
  $("#capture-progress").value = 0
  let lastVideoTime = -1
  return new Promise((resolve, reject) => {
    function frame(now) {
      const elapsed = now - start
      if (elapsed >= duration) {
        resolve(frames)
        return
      }
      try {
        const remaining = `${Math.ceil((duration - elapsed) / 1000)}s`
        if ($("#countdown").textContent !== remaining) $("#countdown").textContent = remaining
        $("#capture-progress").value = elapsed
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

function drawArms(points) {
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  context.strokeStyle = "#63f0d9"
  context.fillStyle = context.strokeStyle
  context.lineWidth = 4
  for (const arm of [[11, 13, 15], [12, 14, 16]]) {
    const joints = arm.map((index) => points[index])
    if (joints.some((point) => !point || point.visibility < 0.5)) continue
    context.beginPath()
    context.moveTo(joints[0].x * canvas.width, joints[0].y * canvas.height)
    for (const point of joints.slice(1)) context.lineTo(point.x * canvas.width, point.y * canvas.height)
    context.stroke()
    for (const point of joints) {
      context.beginPath()
      context.arc(point.x * canvas.width, point.y * canvas.height, 5, 0, Math.PI * 2)
      context.fill()
    }
  }
}

async function postRecording(stream, tasks, collecting) {
  const face = []
  $("#capture-step").textContent = "Step 1 of 2 · Face"
  $("#capture-hint").textContent = "Keep your head still and face the camera."
  const phases = [["relax", "Relax your face"], ["smile", "Smile big"], ["brows", "Raise your eyebrows"]]
  for (const [index, [phase, message]] of phases.entries()) {
    face.push(...await capture(3000, message, (now) => {
      const points = tasks.face.detectForVideo(video, now).faceLandmarks[0] ?? []
      drawFace(points)
      return { phase, landmarks: points.map(xyz) }
    }).then((frames) => frames.map((frame) => ({ ...frame, t: frame.t + index * 3 }))))
  }
  $("#capture-step").textContent = "Step 2 of 2 · Arms"
  $("#capture-hint").textContent = "Keep both wrists in view, with your arms straight out in front at shoulder height."
  function readPose(now) {
    const pose = tasks.pose.detectForVideo(video, now).landmarks[0] ?? []
    drawArms(pose)
    return { pose: pose.map((point) => [...xyz(point), point.visibility]), hands: [] }
  }
  await capture(3000, "Raise both arms, palms up", readPose)
  const arms = await capture(10000, "Hold both arms steady", readPose)
  stream.getTracks().forEach((track) => track.stop())
  $("#capture").hidden = true
  status.textContent = collecting ? "Saving recording…" : "Checking face and arm movement…"
  const data = new FormData()
  data.set("meta", JSON.stringify({
    name: $("#name").value.trim(), user_agent: navigator.userAgent,
    glasses: $("#glasses").checked, lighting: $("#lighting").value,
  }))
  data.set("face", new Blob([JSON.stringify({ fps: face.length / 9, frames: face })], { type: "application/json" }), "face.json")
  data.set("arms", new Blob([JSON.stringify({ fps: arms.length / 10, frames: arms })], { type: "application/json" }), "arms.json")
  if (collecting) data.set("label", $("#label").value)
  const response = await fetch(collecting ? "/api/collect" : "/api/check", { method: "POST", body: data })
  const result = await response.json().catch(() => null)
  if (!response.ok || !result) {
    throw new Error(typeof result?.detail === "string" ? result.detail : `The recording could not be submitted (HTTP ${response.status}).`)
  }
  if (collecting) status.textContent = `Saved to ${result.path}`
  else {
    status.textContent = ""
    renderResult(result)
  }
}

function renderResult(result) {
  const panel = $("#result")
  panel.hidden = false
  panel.dataset.outcome = result.overall
  $("#features").textContent = JSON.stringify(result.features, null, 2)
  $("#latency").textContent = JSON.stringify(result.latency_ms, null, 2)
  $("#outcome").textContent = {
    warning: "Possible sign detected. Call 911 now.",
    clear: "No face or arm sign observed",
    retake: "Camera check needs a retake",
  }[result.overall]
  $("#retake-reason").textContent = result.overall === "retake" ? result.retake_reason ?? "" : ""
  $("#retake").hidden = result.overall === "warning"
  $("#retake").textContent = result.overall === "retake" ? "Retake check" : "Run another check"
  $("#signs").replaceChildren(...Object.entries(result.signs).map(([name, sign]) => {
    const card = document.createElement("article")
    card.className = sign.flag ? "flagged" : "sign"
    const title = document.createElement("h3")
    title.textContent = { face: "Facial asymmetry", arms: "Arm drift" }[name]
    const detail = document.createElement("p")
    detail.textContent = {
      possible_sign: "Possible sign",
      unavailable: "Camera view insufficient",
      not_observed: "Sign not observed",
    }[sign.status]
    card.append(title, detail)
    return card
  }))
  $("#outcome").focus()
  if (result.overall === "warning") startEmergency(result)
}

$("#collect").onchange = (event) => { $("#labels").hidden = !event.target.checked }
$("#setup").onsubmit = async (event) => {
  event.preventDefault()
  if (alertPending) {
    $("#alert-status").focus()
    return
  }
  const collecting = $("#collect").checked
  resetEmergency()
  $("#result").hidden = true
  let stream
  const inputs = [...$("#setup").elements]
  inputs.forEach((input) => { input.disabled = true })
  $(".setup-layout").hidden = true
  $("#contacts").hidden = true
  status.textContent = "Loading camera and models…"
  try {
    visionTasks ??= createTasks().catch((error) => { visionTasks = null; throw error })
    const tasks = await visionTasks
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    video.srcObject = stream
    await video.play()
    $("#capture").hidden = false
    status.textContent = ""
    const recording = postRecording(stream, tasks, collecting)
    $("#prompt").focus()
    await recording
  } catch (error) {
    status.textContent = error.message
  } finally {
    stream?.getTracks().forEach((track) => track.stop())
    video.srcObject = null
    $("#capture").hidden = true
    $("#prompt").textContent = ""
    if ($("#result").hidden) $(".setup-layout").hidden = false
    inputs.forEach((input) => { input.disabled = false })
  }
}

$("#retake").onclick = () => $("#setup").requestSubmit()
$("#back-to-start").onclick = () => {
  if (alertPending) {
    $("#alert-status").focus()
    return
  }
  resetEmergency()
  $("#result").hidden = true
  $(".setup-layout").hidden = false
  $("#start").focus()
}

async function api(path, payload) {
  const options = payload === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  }
  const response = await fetch(path, options)
  const result = await response.json()
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Request failed.")
  return result
}

let contactsUser

function showContacts(contacts) {
  $("#contact-list").replaceChildren(...contacts.map((contact) => {
    const item = document.createElement("li")
    item.textContent = `${contact.name} (${contact.relationship}) · ${contact.email}`
    return item
  }))
}

function deliverySummary(sent) {
  return sent.length ? sent.map((item) => `${item.name} · ${item.channel} · ${item.ok ? "sent" : item.error}`).join("\n") : "No consenting contacts are saved."
}

$("#contacts-open").onclick = async () => {
  if (!$("#name").reportValidity()) return
  contactsUser = encodeURIComponent($("#name").value.trim())
  $("#contacts").hidden = false
  $("#contact-form input").focus()
  $("#contacts-status").textContent = ""
  try {
    showContacts(await api(`/api/contacts/${contactsUser}`))
  } catch (error) {
    $("#contacts-status").textContent = error.message
  }
}
$("#contacts-close").onclick = () => { $("#contacts").hidden = true }
$("#contact-form").onsubmit = async (event) => {
  event.preventDefault()
  const payload = Object.fromEntries(new FormData(event.target))
  payload.consent = payload.consent === "on"
  try {
    showContacts(await api(`/api/contacts/${contactsUser}`, payload))
    event.target.reset()
    $("#contacts-status").textContent = "Contact saved."
  } catch (error) {
    $("#contacts-status").textContent = error.message
  }
}
$("#contacts-test").onclick = async (event) => {
  event.target.disabled = true
  try {
    $("#contacts-status").textContent = "Sending test email…"
    const { sent } = await api(`/api/alert/test/${contactsUser}`, {})
    $("#contacts-status").textContent = deliverySummary(sent)
  } catch (error) {
    $("#contacts-status").textContent = error.message
  } finally {
    event.target.disabled = false
  }
}

let emergencyRun = 0
let agentState
let lastNormalTimer
let alertPending = false

function resetEmergency() {
  emergencyRun += 1
  agentState = undefined
  clearInterval(lastNormalTimer)
  $("#emergency").hidden = true
  $("#agent-say").textContent = ""
  $("#last-normal").textContent = "Last known normal time is unknown."
  $("#agent-answer").reset()
  $("#agent-answer button").disabled = true
  $("#agent-actions").replaceChildren()
  $("#handoff").replaceChildren()
  $("#print-handoff").hidden = true
}

function updateAgent(response, run) {
  if (run !== emergencyRun) return
  agentState = response.state
  $("#agent-answer button").disabled = false
  $("#agent-say").textContent = response.say ?? ""
  const handoff = response.handoff ?? {}
  const facts = handoff.facts ?? {}
  const known = facts.last_known_normal ?? agentState?.timer?.last_known_normal
  const knownAt = typeof known === "string" && /^\d{4}-\d{2}-\d{2}T/.test(known) ? Date.parse(known) : NaN
  clearInterval(lastNormalTimer)
  function timer() {
    if (!known) return
    const elapsed = Math.max(0, Math.floor((Date.now() - knownAt) / 1000))
    const minutes = Math.floor(elapsed / 60)
    $("#last-normal").textContent = Number.isFinite(knownAt)
      ? `Time since last known normal · ${minutes} min ${elapsed % 60} s`
      : `Last known normal · ${known}`
  }
  timer()
  if (Number.isFinite(knownAt)) lastNormalTimer = setInterval(timer, 1000)

  $("#handoff").replaceChildren()
  const entries = { ...handoff.signs, ...facts }
  if (handoff.check_time) entries.check_time = handoff.check_time
  for (const [key, value] of Object.entries(entries)) {
    if (value == null) continue
    const label = document.createElement("dt")
    label.textContent = key.replaceAll("_", " ")
    const detail = document.createElement("dd")
    detail.textContent = typeof value === "object" ? JSON.stringify(value) : value
    $("#handoff").append(label, detail)
  }
  if (agentState?.session) {
    $("#print-handoff").href = `/api/agent/handoff/${encodeURIComponent(agentState.session)}`
    $("#print-handoff").hidden = false
  }
  for (const action of response.actions ?? []) {
    const item = document.createElement("li")
    item.textContent = (action.tool ?? action.name ?? "Action").replaceAll("_", " ")
    $("#agent-actions").append(item)
  }
}

async function startEmergency(result) {
  const run = emergencyRun
  $("#emergency").hidden = false
  const user = $("#name").value.trim()
  startAlertCountdown(user, result)
  $("#agent-say").textContent = "Call 911 now and follow the dispatcher’s instructions."
  try {
    updateAgent(await api("/api/agent/start", { result: { ...result, user }, language: "en" }), run)
  } catch {
    if (run === emergencyRun) $("#agent-say").textContent = "Call 911 now and follow the dispatcher’s instructions. Tell them when the person was last acting normally."
  }
}

$("#agent-answer").onsubmit = async (event) => {
  event.preventDefault()
  if (!agentState) return
  const run = emergencyRun
  const input = $("#family-message")
  const text = input.value.trim()
  if (!text) return
  const button = $("#agent-answer button")
  button.disabled = true
  try {
    updateAgent(await api("/api/agent/message", { text, state: agentState, language: "en" }), run)
    if (run === emergencyRun) input.value = ""
  } catch {
    if (run === emergencyRun) $("#agent-say").textContent = "The assistant could not respond. Follow the 911 dispatcher’s instructions."
  } finally {
    if (run === emergencyRun) button.disabled = false
  }
}

function startAlertCountdown(user, result) {
  const detectedAt = new Date().toISOString()
  const deadline = Date.now() + 20000
  const cancel = $("#cancel-alert")
  const output = $("#alert-status")
  let location
  alertPending = true
  cancel.hidden = false
  if ($("#share-location").checked) {
    navigator.geolocation?.getCurrentPosition(({ coords }) => {
      location = { latitude: coords.latitude, longitude: coords.longitude }
    }, () => {}, { timeout: 10000 })
  }
  cancel.onclick = () => {
    clearInterval(timer)
    alertPending = false
    cancel.hidden = true
    output.textContent = "Trusted contact alerts cancelled."
  }
  async function tick() {
    const remaining = Math.ceil((deadline - Date.now()) / 1000)
    if (remaining > 0) {
      output.textContent = `Emailing trusted contacts in ${remaining} seconds. Keep this page open.`
      return
    }
    clearInterval(timer)
    cancel.hidden = true
    output.textContent = "Sending alerts…"
    try {
      const { sent } = await api("/api/alert", { user, result, detected_at: detectedAt, location, session: agentState?.session })
      output.textContent = `${new Date().toLocaleTimeString()} · ${deliverySummary(sent)}`
    } catch (error) {
      output.textContent = `Alerts could not be sent. ${error.message}`
    } finally {
      alertPending = false
    }
  }
  const timer = setInterval(tick, 250)
  tick()
}
