# call_service.py

import os
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM_NUMBER")  # your Twilio number, e.g. +18001234567
SUPPORT_NUMBER = os.getenv("SUPPORT_NUMBER")   # agent / queue number, e.g. +18007654321


if not ACCOUNT_SID or not AUTH_TOKEN:
    raise RuntimeError("Twilio credentials not set in environment variables.")


client = Client(ACCOUNT_SID, AUTH_TOKEN)


def call_support(customer_number: str | None = None) -> None:
    """
    Initiate a phone call for human assistance.

    - If you want to call a fixed support/queue number, leave `customer_number` as None.
    - If you want to call the customer instead, pass their phone number (E.164 format).
    """
    to_number = SUPPORT_NUMBER if customer_number is None else customer_number

    if not TWILIO_FROM:
        raise RuntimeError("TWILIO_FROM_NUMBER not set in environment.")
    if not to_number:
        raise RuntimeError("SUPPORT_NUMBER (or customer_number) not set.")

    print(f"[DEBUG] Initiating Twilio call from {TWILIO_FROM} to {to_number}...")

    call = client.calls.create(
        from_=TWILIO_FROM,
        to=to_number,
        # Simple inline TwiML: Twilio says this when the call connects
        twiml="<Response><Say>Connecting you to a human agent.</Say></Response>",
    )

    print(f"[DEBUG] Call started. SID: {call.sid}")
