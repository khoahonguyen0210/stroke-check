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
let voiceGeneration = 0
let finishVoice
let listening
let silenceTimer
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
  $("#steps").replaceChildren()
  $("#handoff").replaceChildren()
  $("#model-badge").textContent = "waiting…"
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
  listening = recognition
  recognition.lang = SPEECH_LANG[language()]
  recognition.continuous = false
  recognition.interimResults = true
  $("#mic-indicator").hidden = false
  $("#mic-label").textContent = "Listening…"
  $("#live-transcript").textContent = ""
  $("#talk").hidden = true
  silenceTimer = setTimeout(() => {
    if (listening !== recognition) return
    if (agentState?.paramedics_here) {
      startListening()
      return
    }
    stopListening()
    sendMessage("(no answer)")
  }, 20000)
  recognition.onresult = (event) => {
    if (listening !== recognition) return
    clearTimeout(silenceTimer)
    let interim = ""
    heard = ""
    for (const result of event.results) {
      if (result.isFinal) heard += result[0].transcript
      else interim += result[0].transcript
    }
    $("#live-transcript").textContent = `${heard} ${interim}`.trim()
  }
  recognition.onerror = (event) => {
    if (event.error === "not-allowed" || event.error === "service-not-allowed") {
      micBlocked = true
      $("#mic-label").textContent = "Microphone blocked. Type your answer."
    }
  }
  recognition.onend = () => {
    if (listening !== recognition) return
    const text = heard.trim()
    if (text) {
      stopListening()
      sendMessage(text)
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

function addLine(role, text) {
  if (!text) return
  const item = document.createElement("li")
  item.className = role
  item.textContent = text
  $("#conversation-log").append(item)
  item.scrollIntoView({ block: "nearest" })
}

function shortArgs(args) {
  const text = JSON.stringify(args ?? {})
  return text === "{}" ? "" : text.length > 70 ? `${text.slice(0, 70)}…` : text
}

function renderSteps(steps) {
  const list = $("#steps")
  if (list.children.length) {
    const divider = document.createElement("li")
    divider.className = "turn"
    divider.textContent = "next turn"
    list.append(divider)
  }
  for (const step of steps) {
    const item = document.createElement("li")
    if (step.blocked_by_safety) item.classList.add("blocked")
    if (step.by === "code") item.classList.add("code")
    const name = document.createElement("strong")
    name.textContent = step.tool
    const args = document.createElement("code")
    args.textContent = shortArgs(step.args)
    const result = document.createElement("span")
    result.className = "step-result"
    result.textContent = step.blocked_by_safety ? `blocked by safety → ${step.replaced_with}` : `→ ${step.result}`
    const meta = document.createElement("span")
    meta.className = "step-meta"
    meta.textContent = [step.ms != null ? `${step.ms} ms` : null, step.by === "code" ? "rule in code" : `reasoning ${step.reasoning ? "on" : "off"}`].filter(Boolean).join(" · ")
    item.append(name, " ", args, " ", result, meta)
    list.append(item)
  }
  list.scrollTop = list.scrollHeight
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
  $("#model-badge").textContent = response.model ?? "fixed script"
  renderSteps(response.steps ?? [])
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

const CONFIG_NAMES = {
  bolt_off: "Nemotron 3.5 Bolt · reasoning off",
  bolt_on: "Nemotron 3.5 Bolt · reasoning on",
  super: "Nemotron 3.5 Super",
  gemini: "Gemini 3.5 Flash",
}
const CONFIG_ORDER = ["bolt_off", "bolt_on", "super", "gemini"]

function cell(row, text, header = false) {
  const element = document.createElement(header ? "th" : "td")
  element.textContent = text
  row.append(element)
}

function percent(value) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "–"
}

function renderWhy(results) {
  const runs = results.agent?.runs ?? []
  const versions = [...new Set(runs.map((run) => run.prompt_version))].sort()
  const latest = versions.at(-1)
  const current = {}
  for (const run of runs) if (run.prompt_version === latest) current[run.config] = run
  const configs = CONFIG_ORDER.filter((name) => current[name])
  $("#why-intro").textContent = latest
    ? `Prompt version ${latest}: ${current[configs[0]]?.scenarios ?? 0} scripted emergency conversations, checked automatically. Every number comes from models/results.json.`
    : "No agent evaluation has been saved yet."
  const table = $("#why-table")
  table.replaceChildren()
  const head = document.createElement("tr")
  cell(head, "", true)
  for (const name of configs) cell(head, CONFIG_NAMES[name] ?? name, true)
  table.append(head)
  const rows = [
    ["Scenarios passed", (run) => percent(run.pass_rate)],
    ["Median turn (ms)", (run) => run.turn_latency_ms?.median ?? "–"],
    ["p95 turn (ms)", (run) => run.turn_latency_ms?.p95 ?? "–"],
    ["Safety layer stepped in", (run) => run.safety_interventions ?? "–"],
    ["Unsupported facts before Super check", (run) => run.unsupported_facts_before_super_check ?? "–"],
    ["Unsupported facts after Super check", (run) => run.unsupported_facts_after_super_check ?? "–"],
  ]
  const checks = [...new Set(configs.flatMap((name) => Object.keys(current[name].check_rates ?? {})))]
    .filter((name) => !name.startsWith("script_has_") && !name.startsWith("script_lacks_")).sort()
  for (const check of checks) rows.push([check.replaceAll("_", " "), (run) => percent(run.check_rates?.[check])])
  for (const [label, value] of rows) {
    const row = document.createElement("tr")
    cell(row, label, true)
    for (const name of configs) {
      const run = current[name]
      cell(row, run.valid === false ? "not scored" : value(run))
    }
    table.append(row)
  }
  const chart = $("#why-chart")
  chart.replaceChildren()
  for (const name of CONFIG_ORDER) {
    const scored = runs.filter((run) => run.config === name && run.valid !== false)
    if (!scored.length) continue
    const group = document.createElement("div")
    group.className = "bar-group"
    const title = document.createElement("strong")
    title.textContent = CONFIG_NAMES[name] ?? name
    group.append(title)
    for (const version of versions) {
      const run = scored.filter((item) => item.prompt_version === version).at(-1)
      if (!run) continue
      const bar = document.createElement("div")
      bar.className = "bar"
      const label = document.createElement("span")
      label.textContent = version
      const fill = document.createElement("span")
      fill.className = "bar-fill"
      fill.style.width = `${Math.max(1, run.pass_rate * 100)}%`
      const value = document.createElement("span")
      value.textContent = percent(run.pass_rate)
      bar.append(label, fill, value)
      group.append(bar)
    }
    chart.append(group)
  }
  const unscored = runs.filter((run) => run.valid === false)
  $("#why-notes").textContent = unscored.length
    ? `${unscored.length} runs are not shown as scores because most of their API calls were rejected by rate limits or server errors (${[...new Set(unscored.map((run) => `${run.prompt_version} ${CONFIG_NAMES[run.config] ?? run.config}`))].join(", ")}).`
    : ""
}

for (const button of document.querySelectorAll(".why-open")) {
  button.onclick = async () => {
    $("#why").hidden = false
    $("#why-intro").textContent = "Loading results…"
    try {
      const response = await fetch("/api/benchmarks")
      if (!response.ok) throw new Error()
      renderWhy(await response.json())
    } catch {
      $("#why-intro").textContent = "Results are not available yet."
    }
    $("#why").scrollIntoView({ behavior: "smooth" })
  }
}
$("#why-close").onclick = () => { $("#why").hidden = true }
