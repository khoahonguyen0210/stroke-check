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
      $("#speech-tip").hidden = true
      done.onclick = null
    }
    recorder.ondataavailable = ({ data }) => chunks.push(data)
    recorder.onerror = ({ error }) => { cleanup(); reject(error) }
    recorder.onstop = () => { cleanup(); resolve(new Blob(chunks, { type: recorder.mimeType })) }
    recorder.start()
    done.hidden = false
    $("#speech-tip").hidden = false
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
  data.set("face", new Blob([JSON.stringify({ fps: face.length / 9, frames: face })], { type: "application/json" }), "face.json")
  data.set("arms", new Blob([JSON.stringify({ fps: arms.length / 10, frames: arms })], { type: "application/json" }), "arms.json")
  data.set("audio", audio, "speech.webm")
  if (collecting) data.set("label", $("#label").value)
  else {
    if ($("#gemini-consent").checked) data.set("gemini_consent", "true")
    if ($("#gemini-demo").checked) data.set("gemini_demo", "true")
  }
  const response = await fetch(collecting ? "/api/collect" : "/api/check", { method: "POST", body: data })
  const result = await response.json().catch(() => null)
  if (!response.ok || !result) {
    throw new Error(typeof result?.detail === "string" ? result.detail : `The recording could not be submitted (HTTP ${response.status}).`)
  }
  if (collecting) status.textContent = `Saved to ${result.path}`
  else renderResult(result)
}

function renderResult(result) {
  const geminiDemo = result.overall === "gemini_demo"
  const review = result.gemini_review
  $("#result").hidden = false
  $("#features").textContent = JSON.stringify(result.features, null, 2)
  $("#scores").textContent = JSON.stringify(geminiDemo ? review.signs ?? {} : Object.fromEntries(Object.entries(result.signs).map(([name, sign]) => [name, sign.p])), null, 2)
  $("#latency").textContent = JSON.stringify(result.latency_ms, null, 2)
  $("#outcome").textContent = {
    warning: "Possible stroke signs. Call 911 now.",
    clear: "No warning signs detected by this check",
    retake: "Recording needs a retake",
    gemini_demo: "Gemini demo result",
  }[result.overall]
  $("#retake-reason").textContent = result.retake_reason ?? ""
  const incomplete = geminiDemo && (
    review.status !== "reviewed" || Object.values(review.signs).includes("insufficient_evidence")
  )
  $("#retake").hidden = result.overall !== "retake" && !incomplete
  $("#signs").replaceChildren(...Object.entries(result.signs).map(([name, sign]) => {
    const card = document.createElement("article")
    card.className = sign.flag ? "flagged" : "sign"
    const title = document.createElement("h3")
    title.textContent = { face: "Face droop", arms: "Arm drift", speech: "Slurred speech" }[name]
    const detail = document.createElement("p")
    const assessment = sign.uncertain ? "Uncertain" : sign.flag ? "Flagged" : "Below threshold"
    detail.textContent = sign.p == null ? "Unavailable" : `${assessment} · ${Math.round(sign.p * 100)}%`
    card.append(title, detail)
    return card
  }))
  $("#gemini-heading").textContent = geminiDemo ? "Per-sign ratings" : "Gemini second review"
  $("#gemini-review").hidden = !review
  $("#gemini-signs").replaceChildren()
  if (review) {
    const labels = { face: "Face droop", arms: "Arm drift", speech: "Slurred speech" }
    const observations = {
      possible_sign: "Possible sign. If you suspect a stroke, call 911 now.",
      insufficient_evidence: "Insufficient evidence",
      sign_not_observed: "Sign not observed in this review",
    }
    $("#gemini-summary").textContent = review.status === "reviewed"
      ? (geminiDemo ? "Experimental ratings from this recording." : "Experimental observations. The recording still needs a retake.")
      : (review.status === "poor_quality" ? "Recording quality was too low for a review." : "Gemini review is unavailable.")
    for (const [name, observation] of Object.entries(review.signs ?? {})) {
      const item = document.createElement("li")
      item.textContent = `${labels[name]} · ${observations[observation]}`
      $("#gemini-signs").append(item)
    }
  }
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
  if (!collecting && $("#gemini-demo").checked && !$("#gemini-consent").checked) {
    status.textContent = "Allow Google Gemini to receive the speech audio and measurements before starting the Gemini-only demo."
    $("#gemini-consent").focus()
    return
  }
  resetEmergency()
  $("#result").hidden = true
  let stream
  const inputs = [...$("#setup").elements]
  inputs.forEach((input) => { input.disabled = true })
  status.textContent = "Loading camera and models…"
  try {
    visionTasks ??= createTasks().catch((error) => { visionTasks = null; throw error })
    const tasks = await visionTasks
    stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: { noiseSuppression: true, echoCancellation: true, channelCount: 1 },
    })
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

$("#retake").onclick = () => $("#setup").requestSubmit()

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
    $("#contacts-status").textContent = "Sending test alert…"
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
  window.speechSynthesis?.cancel()
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
  if (response.say && window.speechSynthesis) {
    window.speechSynthesis.cancel()
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(response.say))
  }

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
      output.textContent = `Alerting your trusted contacts in ${remaining} seconds.`
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


$("#benchmark").ontoggle = async (event) => {
  if (!event.target.open) return
  const output = $("#benchmark-data")
  output.textContent = "Loading benchmark results…"
  try {
    output.textContent = JSON.stringify(await api("/api/benchmarks"), null, 2)
  } catch (error) {
    output.textContent = error.message
  }
}
