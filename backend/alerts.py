import json
import os
import smtplib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.message import EmailMessage
from threading import Lock
from urllib.parse import urlencode

from dotenv import load_dotenv

from backend import storage

DISCLAIMER = "This is an automated warning, not a diagnosis."
LOCK = Lock()


def deliver(user, subject, text):
    load_dotenv(storage.DATA.parent / ".env")
    sender = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_APP_PASSWORD")

    def email(contact):
        result = {"name": contact["name"], "channel": "email", "ok": False}
        if not sender or not password:
            return {**result, "error": "Email is not configured."}
        message = EmailMessage()
        message["From"] = sender
        message["To"] = contact["email"]
        message["Subject"] = subject
        message.set_content(text)
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=8) as connection:
                connection.login(sender, password)
                connection.send_message(message)
            return {**result, "ok": True}
        except (OSError, smtplib.SMTPException):
            return {**result, "error": "Email delivery failed."}

    recipients = [contact for contact in storage.contacts(user) if contact.get("consent") is True]
    with ThreadPoolExecutor(max_workers=3) as pool:
        sent = list(pool.map(email, recipients))
    with LOCK:
        storage.DATA.mkdir(parents=True, exist_ok=True)
        with (storage.DATA / "alerts.log").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"time": datetime.now().astimezone().isoformat(), "user": user, "subject": subject, "sent": sent}) + "\n")
    return sent


def send(user, result, detected_at, location=None):
    names = {"face": "face droop", "arms": "arm drift", "speech": "slurred speech"}
    signs = ", ".join(label for sign, label in names.items() if result.get("signs", {}).get(sign, {}).get("flag") is True)
    if not signs:
        return []
    where = "not shared"
    if location:
        latitude, longitude = location.get("latitude"), location.get("longitude")
        if isinstance(latitude, (float, int)) and isinstance(longitude, (float, int)) and -90 <= latitude <= 90 and -180 <= longitude <= 180:
            where = "https://www.google.com/maps/search/?" + urlencode({"api": 1, "query": f"{latitude},{longitude}"})
    message = (
        f"{user}'s Stroke Check found possible stroke signs ({signs}) at {detected_at}. "
        "They have been told to call 911. Please call them now or call 911 if you can't reach them. "
        f"Location: {where}. {DISCLAIMER}"
    )
    return deliver(user, "Stroke Check warning", message)


def notify(user, text):
    return deliver(user, "Stroke Check update", f"{text}\nFollow the 911 dispatcher's instructions. {DISCLAIMER}")


def test(user):
    return deliver(user, "Test alert from Stroke Check", f"This is a test. In a real emergency, call 911. {DISCLAIMER}")
