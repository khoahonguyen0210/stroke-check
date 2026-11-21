import re

REASSURANCE = [
    r"\b(you|he|she|they|it|everything)\s*(are|is|'re|'s|re|s)?\s*(fine|ok|okay|all right|alright)\b(?!\s+(with|to|if|for)\b)",
    r"\b(nothing|no need) to worry\b",
    r"\bprobably nothing\b",
    r"\bdon'?t worry\b",
    r"\b(usted|él|ella|todo)\s+(está|estás|esta)\s+bien\b",
    r"\bestás bien\b",
    r"\bno es nada\b",
    r"\bno (te|se) preocupe",
    r"\bkhông (sao|có gì) (đâu|cả|nghiêm trọng)",
    r"\b(bạn|anh|chị|ông|bà|em|cô|chú|mọi thứ)\s+(vẫn\s+)?ổn\b",
    r"\bđừng lo\b",
]

NOT_STROKE = [
    r"\b(not|isn'?t|is not|wasn'?t|nothing like) (a |an )?stroke\b",
    r"\bno (es|fue) un (derrame|ictus|accidente cerebrovascular)",
    r"\bkhông phải (là )?đột quỵ",
]

WAIT = [
    r"\bwait and see\b",
    r"\bwait (a |an )?(few|bit|while|little|hour|minute|until|to see)",
    r"\b(see|check) if (it|they|he|she|the symptoms?) (gets?|go(es)?|improves?)",
    r"\b(no need|don'?t need|do not need|not necessary) to call\b",
    r"\b(don'?t|do not|no need to) call (911|an ambulance|for help)",
    r"\bdrive (him|her|them) (yourself|to the hospital)",
    r"\bespera(r|mos|n)? a ver\b",
    r"\bno (hace falta|es necesario|necesita) llamar",
    r"\bno llame",
    r"\bchờ (xem|thêm|một (chút|lúc))",
    r"\bkhông cần (phải )?gọi",
    r"\bđừng gọi",
]

DIAGNOSIS = [
    r"\b(you|he|she|they|he's|she's|they're|you're)\s*(is|are|has|have|'s|'re)?\s*(having|had|has|have|suffering)\s+(a |an )?(stroke|tia|mini-?stroke|bell'?s palsy|seizure|brain bleed)",
    r"\b(this|it|that) (is|was|'s) (definitely |clearly |probably |just )?(a |an )?(stroke|tia|mini-?stroke|bell'?s palsy|migraine|seizure|brain bleed)\b",
    r"\b(definitely|certainly|clearly|confirmed) (a |an )?stroke\b",
    r"\bdiagnos(e|is|ed|ing)\b",
    r"\b(it'?s|it is|that'?s) (just|only) (stress|tiredness|fatigue|anxiety|dehydration|low blood sugar)",
    r"\b(tiene|tuvo|sufre|sufrió) un (derrame|ictus|accidente cerebrovascular)",
    r"\b(es|fue) (un|una) (derrame|ictus|accidente cerebrovascular|parálisis de bell)",
    r"\bdiagn[oó]stic",
    r"\b(đang|đã) bị đột quỵ",
    r"\b(chắc chắn|rõ ràng) (là )?(bị )?đột quỵ",
    r"\bchẩn đoán",
]

MEDICINE_WORDS = (
    r"aspirin|aspirina|ibuprofen|ibuprofeno|advil|motrin|tylenol|acetaminophen|paracetamol|nitroglycerin|"
    r"medicine|medication|meds|pill|pills|tablet|tablets|dose|drug|drugs|"
    r"medicina|medicamento|medicamentos|pastilla|pastillas|dosis|"
    r"thuốc|liều|viên"
)
DOSE = r"\b\d+(\.\d+)?\s*(mg|milligrams?|ml|mcg|units?|tablets?|pills?|viên|miligramos?)\b"
NEGATION = r"\b(no|not|don'?t|do not|never|avoid|without|nunca|evite|evita|sin|không|đừng|chớ|tránh)\b"
QUESTION_START = r"^\s*(¿|are|is|do|does|did|has|have|what|which|any|toma|tiene|usa|ông|bà|anh|chị|bạn|họ|có)\b"

FIXED_SCRIPT = {
    "en": [
        "Possible stroke signs were detected. This is a warning, not a diagnosis.",
        "Call 911 now.",
        "Note the time you last saw them acting normal.",
        "Do not give any food, drink or medicine.",
        "Stay with them and unlock the door for the paramedics.",
    ],
    "es": [
        "Se detectaron posibles señales de derrame cerebral. Es una advertencia, no un diagnóstico.",
        "Llame al 911 ahora.",
        "Anote la hora en que lo vio normal por última vez.",
        "No le dé comida, bebida ni medicinas.",
        "Quédese con la persona y abra la puerta para los paramédicos.",
    ],
    "vi": [
        "Đã phát hiện dấu hiệu có thể là đột quỵ. Đây là cảnh báo, không phải chẩn đoán.",
        "Hãy gọi 911 ngay.",
        "Ghi lại thời điểm cuối cùng bạn thấy người đó còn bình thường.",
        "Không cho ăn, uống hay dùng bất kỳ thuốc nào.",
        "Ở bên cạnh người đó và mở khóa cửa cho nhân viên cấp cứu.",
    ],
}

REPLACEMENT = {
    "reassurance": {
        "en": "Only a doctor can tell what is happening. Call 911 now.",
        "es": "Solo un médico puede saber qué está pasando. Llame al 911 ahora.",
        "vi": "Chỉ bác sĩ mới biết chuyện gì đang xảy ra. Hãy gọi 911 ngay.",
    },
    "not_stroke": {
        "en": "Only a doctor can tell what is happening. Call 911 now.",
        "es": "Solo un médico puede saber qué está pasando. Llame al 911 ahora.",
        "vi": "Chỉ bác sĩ mới biết chuyện gì đang xảy ra. Hãy gọi 911 ngay.",
    },
    "wait": {
        "en": "Do not wait, even if the signs seem to go away. Call 911 now.",
        "es": "No espere, aunque las señales parezcan desaparecer. Llame al 911 ahora.",
        "vi": "Đừng chờ, kể cả khi các dấu hiệu có vẻ hết. Hãy gọi 911 ngay.",
    },
    "medicine": {
        "en": "Do not give any medicine, food or drink. The paramedics will decide.",
        "es": "No le dé medicinas, comida ni bebida. Los paramédicos decidirán.",
        "vi": "Không cho dùng thuốc, ăn hay uống. Nhân viên cấp cứu sẽ quyết định.",
    },
    "diagnosis": {
        "en": "These are warning signs, not a diagnosis. Call 911 now.",
        "es": "Son señales de advertencia, no un diagnóstico. Llame al 911 ahora.",
        "vi": "Đây là dấu hiệu cảnh báo, không phải chẩn đoán. Hãy gọi 911 ngay.",
    },
}


def _normalize(text):
    return re.sub(r"\s+", " ", text.lower().replace("’", "'").replace("‘", "'")).strip()


def _any(patterns, text):
    return any(re.search(p, text) for p in patterns)


def _medicine_advice(text):
    if re.search(DOSE, text):
        return True
    for sentence in re.split(r"(?<=[.!?;])\s+", text):
        if not re.search(rf"\b({MEDICINE_WORDS})\b", sentence):
            continue
        if re.search(NEGATION, sentence):
            continue
        if sentence.rstrip().endswith("?") and re.search(QUESTION_START, sentence):
            continue
        return True
    return False


def check(text):
    t = _normalize(text or "")
    if not t:
        return {"ok": False, "reason": "empty"}
    if _any(NOT_STROKE, t):
        return {"ok": False, "reason": "not_stroke"}
    if _any(WAIT, t):
        return {"ok": False, "reason": "wait"}
    if _any(DIAGNOSIS, t):
        return {"ok": False, "reason": "diagnosis"}
    if _medicine_advice(t):
        return {"ok": False, "reason": "medicine"}
    if _any(REASSURANCE, t):
        return {"ok": False, "reason": "reassurance"}
    return {"ok": True, "reason": None}


def replacement(reason, language="en"):
    lang = language if language in FIXED_SCRIPT else "en"
    if reason in REPLACEMENT:
        return REPLACEMENT[reason][lang]
    return FIXED_SCRIPT[lang][1]


def fixed_script(language="en"):
    return list(FIXED_SCRIPT.get(language, FIXED_SCRIPT["en"]))
