const $ = (selector) => document.querySelector(selector)
const incidentId = decodeURIComponent(location.pathname.split("/").filter(Boolean).pop() || "")
const STYLES = {
  light: "https://tiles.openfreemap.org/styles/liberty",
  dark: "https://tiles.openfreemap.org/styles/dark",
}
const FIELD_NAMES = {
  blood_thinners: "Blood thinners",
  recent_fall_or_head_injury: "Recent fall or head injury",
  diabetes: "Diabetes",
  seizure: "Seizure",
  current_medications: "Current medications",
  allergies: "Allergies",
}
const SIGN_NAMES = { face: "face droop", arms: "arm drift", speech: "slurred speech" }

let theme = window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light"
let follow = true
let personMarker
let personAt
let meMarker
let meAt
let hospitalMarkers = new Map()
let polling

const map = new maplibregl.Map({
  container: "map",
  style: STYLES[theme],
  center: [-98.5, 39.8],
  zoom: 3,
  pitch: 45,
  attributionControl: { compact: true },
})
map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "bottom-right")

function viewPadding() {
  const sheet = $("#sheet")
  return window.innerWidth >= 820
    ? { top: 70, right: sheet.offsetWidth + 20, bottom: 40, left: 40 }
    : { top: 70, right: 20, bottom: sheet.offsetHeight + 20, left: 20 }
}
map.on("load", () => map.setPadding(viewPadding()))
window.addEventListener("resize", () => map.setPadding(viewPadding()))
map.on("dragstart", () => {
  follow = false
  $("#recenter").hidden = !personAt
})

function applyTheme() {
  document.body.classList.toggle("dark", theme === "dark")
  $("#theme").textContent = theme === "dark" ? "Light map" : "Dark map"
  $("#theme").setAttribute("aria-pressed", String(theme === "dark"))
}
applyTheme()
$("#theme").onclick = () => {
  theme = theme === "dark" ? "light" : "dark"
  map.setStyle(STYLES[theme])
  applyTheme()
}

$("#recenter").onclick = () => {
  follow = true
  $("#recenter").hidden = true
  if (personAt) map.flyTo({ center: [personAt.lng, personAt.lat], zoom: 15, pitch: 50, duration: 1600 })
}

function element(className, text) {
  const node = document.createElement("div")
  node.className = className
  if (text) node.textContent = text
  return node
}

function animateMarker(marker, from, to, duration = 1500) {
  const start = performance.now()
  function frame(now) {
    const t = Math.min(1, (now - start) / duration)
    const ease = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2
    marker.setLngLat([from.lng + (to.lng - from.lng) * ease, from.lat + (to.lat - from.lat) * ease])
    if (t < 1) requestAnimationFrame(frame)
  }
  requestAnimationFrame(frame)
}

function km(a, b) {
  const rad = (d) => (d * Math.PI) / 180
  const h = Math.sin(rad(b.lat - a.lat) / 2) ** 2 + Math.cos(rad(a.lat)) * Math.cos(rad(b.lat)) * Math.sin(rad(b.lng - a.lng) / 2) ** 2
  return 2 * 6371 * Math.asin(Math.sqrt(h))
}

function directions(lat, lng) {
  return `https://www.google.com/maps/dir/?api=1&destination=${lat},${lng}&travelmode=driving`
}

function time(iso) {
  return new Date(iso).toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" })
}

function updatePerson(data) {
  const point = data.location
  if (!point) {
    $("#where").textContent = "Location not shared"
    $("#directions-person").hidden = true
    return
  }
  const next = { lat: point.lat, lng: point.lng }
  if (!personMarker) {
    const pin = element("person-pin")
    pin.append(element("pulse"), element("dot"), element("pin-label", data.name))
    personMarker = new maplibregl.Marker({ element: pin }).setLngLat([next.lng, next.lat]).addTo(map)
    map.setPadding(viewPadding())
    map.flyTo({ center: [next.lng, next.lat], zoom: 15, pitch: 50, bearing: -12, duration: 2600, essential: true })
  } else if (personAt.lat !== next.lat || personAt.lng !== next.lng) {
    animateMarker(personMarker, personAt, next)
    if (follow) map.easeTo({ center: [next.lng, next.lat], duration: 1500 })
  }
  personAt = next
  const distance = meAt ? ` · ${(km(meAt, next) * 0.621371).toFixed(1)} mi from you (straight line)` : ""
  $("#where").textContent = `Location updated ${time(point.at)}${distance}`
  $("#directions-person").href = directions(next.lat, next.lng)
  $("#directions-person").hidden = false
}

function updateHospitals(data) {
  const hospitals = data.hospitals ?? []
  $("#hospital-note").hidden = !hospitals.length
  const destination = data.destination_hospital
  for (const hospital of hospitals) {
    if (hospitalMarkers.has(hospital.name)) continue
    const node = element("hospital")
    node.append(element("hospital-icon", "H"), element("hospital-name", hospital.name))
    hospitalMarkers.set(hospital.name, new maplibregl.Marker({ element: node, anchor: "left" }).setLngLat([hospital.lng, hospital.lat]).addTo(map))
  }
  for (const [name, marker] of hospitalMarkers) {
    marker.getElement().classList.toggle("destination", Boolean(destination && destination.name === name && destination.lat != null))
  }
  $("#destination").hidden = !destination
  if (destination) {
    $("#destination-text").textContent = `Paramedics are going to ${destination.name}`
    const located = destination.lat != null
    $("#directions-hospital").hidden = !located
    if (located) {
      $("#directions-hospital").href = directions(destination.lat, destination.lng)
      $("#directions-hospital").textContent = `Directions to ${destination.name}`
      if (!updateHospitals.flown) {
        updateHospitals.flown = true
        const points = [[destination.lng, destination.lat], ...(personAt ? [[personAt.lng, personAt.lat]] : [])]
        const bounds = points.reduce((b, p) => b.extend(p), new maplibregl.LngLatBounds(points[0], points[0]))
        map.fitBounds(bounds, { padding: viewPadding(), maxZoom: 15, pitch: 50, duration: 2200 })
      }
    }
  }
}

function updateTimeline(data) {
  const list = $("#timeline")
  list.replaceChildren(...data.status_timeline.map((item) => {
    const row = document.createElement("li")
    const when = document.createElement("time")
    when.textContent = time(item.t)
    const what = document.createElement("span")
    what.textContent = item.event
    row.append(when, what)
    return row
  }))
}

function updateHandoff(data) {
  const box = $("#handoff")
  const handoff = data.handoff
  if (!handoff) return
  const list = document.createElement("dl")
  const add = (label, value) => {
    const dt = document.createElement("dt")
    dt.textContent = label
    const dd = document.createElement("dd")
    dd.textContent = value
    list.append(dt, dd)
  }
  if (handoff.signs?.length) add("Possible signs", handoff.signs.map((s) => SIGN_NAMES[s] ?? s).join(", "))
  if (handoff.time_noticed) add("Last seen normal", handoff.time_noticed)
  for (const [key, value] of Object.entries(handoff.facts ?? {})) if (FIELD_NAMES[key]) add(FIELD_NAMES[key], value)
  const parts = [list]
  if (handoff.verified) {
    const badge = document.createElement("p")
    badge.className = "verified"
    badge.textContent = "Checked by Nemotron Super ✓"
    parts.unshift(badge)
  }
  box.replaceChildren(...parts)
}

function render(data) {
  document.title = `${data.name} · Stroke Check live view`
  const latest = data.status_timeline.at(-1)?.event
  $("#headline").textContent = `Possible stroke signs for ${data.name}${latest ? `. ${latest}.` : "."}`
  if (data.phone) {
    $("#call").href = `tel:${data.phone}`
    $("#call").textContent = `Call ${data.name}`
    $("#call").hidden = false
  }
  updatePerson(data)
  updateHospitals(data)
  updateTimeline(data)
  updateHandoff(data)
}

function expire() {
  clearInterval(polling)
  $("#expired").hidden = false
}

async function refresh() {
  try {
    const response = await fetch(`/api/incident/${encodeURIComponent(incidentId)}`, { cache: "no-store" })
    if (response.status === 404) return expire()
    if (response.ok) render(await response.json())
  } catch {}
}

$("#locate-me").onclick = () => {
  if (!navigator.geolocation) {
    $("#locate-me").textContent = "Location unavailable"
    return
  }
  $("#locate-me").disabled = true
  $("#locate-me").textContent = "Finding you…"
  navigator.geolocation.watchPosition(({ coords }) => {
    meAt = { lat: coords.latitude, lng: coords.longitude }
    if (!meMarker) {
      const dot = element("me-dot")
      meMarker = new maplibregl.Marker({ element: dot }).setLngLat([meAt.lng, meAt.lat]).addTo(map)
    } else {
      meMarker.setLngLat([meAt.lng, meAt.lat])
    }
    $("#locate-me").textContent = "Showing your location"
    if (personAt) {
      const distance = (km(meAt, personAt) * 0.621371).toFixed(1)
      $("#where").textContent = $("#where").textContent.split(" · ")[0] + ` · ${distance} mi from you (straight line)`
    }
  }, () => {
    $("#locate-me").disabled = false
    $("#locate-me").textContent = "Location permission denied"
  }, { enableHighAccuracy: true, maximumAge: 5000 })
}

refresh()
polling = setInterval(refresh, 3000)
