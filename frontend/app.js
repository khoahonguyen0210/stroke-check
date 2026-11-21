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
    warning: "Possible stroke sign. Call 911 now.",
    clear: "Face and arm movements looked typical",
    retake: "Let’s try the camera check again",
  }[result.overall]
  $("#retake-reason").textContent = result.overall === "retake" ? result.retake_reason ?? "" : ""
  $("#retake").hidden = result.overall === "warning"
  $("#retake").textContent = result.overall === "retake" ? "Try again" : "Run another check"
  $("#signs").replaceChildren(...Object.entries(result.signs).map(([name, sign]) => {
    const card = document.createElement("article")
    card.className = sign.flag ? "flagged" : "sign"
    const title = document.createElement("h3")
    title.textContent = { face: "Face movement", arms: "Arm movement" }[name]
    const detail = document.createElement("p")
    detail.textContent = {
      possible_sign: "Possible warning sign",
      unavailable: "The camera couldn’t get a clear view",
      not_observed: "Looked typical in this check",
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
let voiceGeneration = 0
let finishVoice
let listening
let silenceTimer
let submitVoice
const VOICE_PAUSE_MS = 1200
const VOICE_FINALIZE_MS = 700
const NO_ANSWER_MS = 20000
let micBlocked = false
let busy = false
let incidentId
let locationWatch
let lastLocationPost = 0
const SpeechInput = window.SpeechRecognition || window.webkitSpeechRecognition
const SPEECH_LANG = { en: "en-US", es: "es-US", vi: "vi-VN" }
const language = () => $("#agent-language").value

function resetEmergency() {
  emergencyRun += 1
  agentState = undefined
  incidentId = undefined
  if (locationWatch !== undefined) navigator.geolocation?.clearWatch(locationWatch)
  locationWatch = undefined
  busy = false
  stopListening()
  stopVoice()
  clearInterval(lastNormalTimer)
  $("#emergency").hidden = true
  $("#agent-say").textContent = ""
  $("#last-normal").textContent = "Last known normal time is unknown."
  $("#family-message").value = ""
  $("#conversation-log").replaceChildren()
  $("#handoff").replaceChildren()
  $("#script-card").hidden = true
  $("#verified-badge").hidden = true
  $("#removed-facts").hidden = true
  $("#print-handoff").hidden = true
  $("#replay-agent").hidden = true
  $("#talk").hidden = true
  $("#handoff-card").classList.remove("fullscreen")
  $("#close-handoff").hidden = true
  $("#paramedics-here").disabled = false
}

function stopVoice() {
  voiceGeneration += 1
  finishVoice?.()
  window.speechSynthesis?.cancel()
}

function playBrowserVoice(text) {
  if (!window.speechSynthesis) return Promise.resolve()
  return new Promise((resolve) => {
    const utterance = new SpeechSynthesisUtterance(text.replaceAll("911", "nine one one"))
    const end = () => {
      clearTimeout(timeout)
      if (finishVoice === end) finishVoice = undefined
      resolve()
    }
    const timeout = setTimeout(() => {
      window.speechSynthesis.cancel()
      end()
    }, 45000)
    finishVoice = end
    utterance.lang = SPEECH_LANG[language()]
    utterance.onend = end
    utterance.onerror = end
    window.speechSynthesis.speak(utterance)
  })
}

function playClip(blob) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(blob)
    const audio = new Audio(url)
    const end = (played = true) => {
      clearTimeout(timeout)
      audio.pause()
      URL.revokeObjectURL(url)
      if (finishVoice === end) finishVoice = undefined
      resolve(played)
    }
    const timeout = setTimeout(() => end(false), 45000)
    finishVoice = end
    audio.onended = () => end()
    audio.onerror = () => end(false)
    audio.play().catch(() => end(false))
  })
}

async function speakThenListen(text, session) {
  stopListening()
  stopVoice()
  const generation = voiceGeneration
  if (text) {
    const clip = session && language() === "en"
      ? await fetch(`/api/agent/voice/${encodeURIComponent(session)}`, { signal: AbortSignal.timeout?.(4000) })
        .then((response) => response.ok ? response.blob() : null).catch(() => null)
      : null
    if (generation !== voiceGeneration) return
    const played = clip && await playClip(clip)
    if (!played && generation === voiceGeneration) await playBrowserVoice(text)
  }
  if (generation === voiceGeneration && agentState) startListening()
}

$("#replay-agent").onclick = () => speakThenListen($("#agent-say").textContent, agentState?.session)

function stopListening() {
  clearTimeout(silenceTimer)
  const recognition = listening
  listening = undefined
  submitVoice = undefined
  $("#send-voice").hidden = true
  $("#send-voice").disabled = false
  recognition?.abort()
  $("#mic-indicator").hidden = true
  $("#talk").hidden = !(SpeechInput && agentState)
}

function startListening() {
  if (!SpeechInput || micBlocked || busy) {
    $("#talk").hidden = !SpeechInput
    return
  }
  stopListening()
  const recognition = new SpeechInput()
  let heard = ""
  let finishing = false
  listening = recognition
  recognition.lang = SPEECH_LANG[language()]
  recognition.continuous = false
  recognition.interimResults = true
  $("#mic-indicator").hidden = false
  $("#mic-label").textContent = "Listening…"
  $("#live-transcript").textContent = ""
  $("#talk").hidden = true
  $("#send-voice").hidden = false

  function sendHeard() {
    if (listening !== recognition) return
    stopListening()
    if (heard) sendMessage(heard)
  }

  submitVoice = () => {
    if (listening !== recognition || finishing) return
    finishing = true
    $("#mic-label").textContent = "Finishing voice reply…"
    $("#send-voice").disabled = true
    clearTimeout(silenceTimer)
    silenceTimer = setTimeout(sendHeard, VOICE_FINALIZE_MS)
    try {
      recognition.stop()
    } catch {
      sendHeard()
    }
  }

  silenceTimer = setTimeout(() => {
    if (listening !== recognition) return
    if (agentState?.paramedics_here) {
      startListening()
      return
    }
    stopListening()
    sendMessage("(no answer)")
  }, NO_ANSWER_MS)
  recognition.onresult = (event) => {
    if (listening !== recognition) return
    heard = Array.from(event.results, (result) => result[0].transcript.trim()).join(" ").trim()
    $("#live-transcript").textContent = heard
    if (!finishing && heard) {
      clearTimeout(silenceTimer)
      silenceTimer = setTimeout(submitVoice, VOICE_PAUSE_MS)
    }
  }
  recognition.onerror = (event) => {
    if (event.error === "not-allowed" || event.error === "service-not-allowed") {
      micBlocked = true
      $("#mic-label").textContent = "Microphone blocked. Type your answer."
    }
  }
  recognition.onend = () => {
    if (listening !== recognition) return
    if (heard || finishing) {
      sendHeard()
      return
    }
    if (micBlocked) {
      stopListening()
      return
    }
    try {
      recognition.start()
    } catch {
      stopListening()
    }
  }
  try {
    recognition.start()
  } catch {
    stopListening()
  }
}

$("#talk").onclick = () => {
  micBlocked = false
  stopVoice()
  startListening()
}

$("#send-voice").onclick = () => submitVoice?.()
document.addEventListener("keydown", (event) => {
  if (!submitVoice || event.defaultPrevented || event.isComposing || event.repeat) return
  if (event.key !== " " && event.key !== "Enter") return
  if (event.target.closest?.("input, textarea, select, button, a, [contenteditable]")) return
  event.preventDefault()
  submitVoice()
})

function addLine(role, text) {
  if (!text) return
  const item = document.createElement("li")
  item.className = role
  item.textContent = text
  $("#conversation-log").append(item)
  item.scrollIntoView({ block: "nearest" })
}

function renderHandoff(response) {
  const handoff = response.handoff ?? {}
  const facts = handoff.facts ?? {}
  const known = facts.last_known_normal
  $("#last-normal").textContent = known ? `Last known normal (family's words): ${known}` : "Last known normal time is unknown."
  $("#handoff").replaceChildren()
  const entries = {}
  for (const [name, p] of Object.entries(handoff.signs ?? {})) entries[`sign: ${name}`] = typeof p === "number" ? `possible (p = ${p.toFixed(2)})` : "possible"
  if (handoff.someone_present) entries["someone present"] = handoff.someone_present
  if (handoff.called_911) entries["911 called"] = handoff.called_911
  for (const [key, value] of Object.entries(facts)) if (value != null && key !== "last_known_normal") entries[key] = value
  if (handoff.check_time) entries["check time"] = handoff.check_time
  for (const [key, value] of Object.entries(entries)) {
    const label = document.createElement("dt")
    label.textContent = key.replaceAll("_", " ")
    const detail = document.createElement("dd")
    detail.textContent = value
    $("#handoff").append(label, detail)
  }
  $("#verified-badge").hidden = response.verified !== true
  const removed = response.removed_facts ?? []
  $("#removed-facts").hidden = !removed.length
  $("#removed-facts").textContent = removed.length
    ? `Removed by the Nemotron Super check (no supporting line): ${removed.map((fact) => `${fact.field.replaceAll("_", " ")} = ${fact.value}`).join("; ")}`
    : ""
  if (response.verified === false) {
    $("#removed-facts").hidden = false
    $("#removed-facts").textContent = "The Nemotron Super check was unavailable. This handoff was not double-checked."
  }
  if (agentState?.session) {
    $("#print-handoff").href = `/api/agent/handoff/${encodeURIComponent(agentState.session)}`
    $("#print-handoff").hidden = false
  }
}

function updateAgent(response, run) {
  if (run !== emergencyRun) return
  agentState = response.state
  addLine("agent", response.say)
  $("#agent-say").textContent = response.say ?? ""
  $("#replay-agent").hidden = !response.say
  $("#script-card").hidden = !response.script_911
  $("#script-911").textContent = response.script_911 ?? ""
  renderHandoff(response)
  speakThenListen(response.say, agentState?.session)
}

async function sendMessage(text) {
  if (!agentState || busy || !text) return
  busy = true
  stopListening()
  stopVoice()
  const run = emergencyRun
  addLine("family", text)
  $("#mic-indicator").hidden = false
  $("#mic-label").textContent = "Nemotron is thinking…"
  $("#live-transcript").textContent = ""
  try {
    const response = await api("/api/agent/message", { text, state: { session: agentState.session }, language: language() })
    busy = false
    if (run === emergencyRun) updateAgent(response, run)
  } catch {
    busy = false
    if (run === emergencyRun) {
      $("#mic-indicator").hidden = true
      $("#agent-say").textContent = "The assistant could not respond. Call 911 now and follow the dispatcher’s instructions."
    }
  }
}

$("#agent-reply").onsubmit = (event) => {
  event.preventDefault()
  const input = $("#family-message")
  const text = input.value.trim()
  input.value = ""
  sendMessage(text)
}

$("#paramedics-here").onclick = async () => {
  $("#paramedics-here").disabled = true
  await sendMessage("paramedics are here")
  $("#handoff-card").classList.add("fullscreen")
  $("#close-handoff").hidden = false
}

$("#close-handoff").onclick = () => {
  $("#handoff-card").classList.remove("fullscreen")
  $("#close-handoff").hidden = true
}

function getLocation() {
  if (!$("#share-location").checked || !navigator.geolocation) return Promise.resolve(undefined)
  return new Promise((resolve) => {
    const timeout = setTimeout(() => resolve(undefined), 3000)
    navigator.geolocation.getCurrentPosition(({ coords }) => {
      clearTimeout(timeout)
      resolve({ latitude: coords.latitude, longitude: coords.longitude, accuracy: coords.accuracy })
    }, () => {
      clearTimeout(timeout)
      resolve(undefined)
    }, { timeout: 3000 })
  })
}

function postLocation(latitude, longitude, accuracy) {
  if (!incidentId) return
  lastLocationPost = Date.now()
  fetch(`/api/incident/${encodeURIComponent(incidentId)}/location`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lat: latitude, lng: longitude, accuracy }),
  }).catch(() => {})
}

function shareLocation(first) {
  if (!incidentId || !$("#share-location").checked || !navigator.geolocation) return
  if (first) postLocation(first.latitude, first.longitude, first.accuracy)
  locationWatch = navigator.geolocation.watchPosition(({ coords }) => {
    if (Date.now() - lastLocationPost >= 10000) postLocation(coords.latitude, coords.longitude, coords.accuracy)
  }, () => {}, { enableHighAccuracy: true, maximumAge: 5000 })
}

async function startEmergency(result) {
  const run = emergencyRun
  $("#emergency").hidden = false
  $("#emergency").scrollIntoView({ block: "start", behavior: "smooth" })
  $("#agent-say").textContent = "Call 911 now."
  const user = $("#name").value.trim()
  const [location, created] = await Promise.all([
    getLocation(),
    api("/api/incident", { user, name: user, phone: $("#phone").value.trim() || null, result }).catch(() => null),
  ])
  if (run !== emergencyRun) return
  incidentId = created?.id
  shareLocation(location)
  startAlertCountdown(user, result, location)
  try {
    const response = await api("/api/agent/start", { result: { ...result, user, name: user, location, incident: incidentId }, language: language() })
    updateAgent(response, run)
  } catch {
    if (run === emergencyRun) $("#agent-say").textContent = "Call 911 now and follow the dispatcher’s instructions."
  }
}

function startAlertCountdown(user, result, location) {
  const detectedAt = new Date().toISOString()
  const deadline = Date.now() + 20000
  const cancel = $("#cancel-alert")
  const output = $("#alert-status")
  alertPending = true
  cancel.hidden = false
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
      const { sent } = await api("/api/alert", { user, result, detected_at: detectedAt, location, session: agentState?.session, incident: incidentId })
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
