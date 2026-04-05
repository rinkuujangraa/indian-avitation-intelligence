"""
notifier.py
-----------
Sends a one-time email alert when the AirLabs API quota is exhausted
or the key becomes invalid, so you can act immediately.

Setup in .env:
    ALERT_EMAIL_FROM=your.email@gmail.com
    ALERT_EMAIL_TO=your.email@gmail.com        (defaults to FROM address)
    ALERT_EMAIL_PASSWORD=your_gmail_app_password
    ALERT_SMTP_HOST=smtp.gmail.com             (optional, default: smtp.gmail.com)
    ALERT_SMTP_PORT=587                        (optional, default: 587)

Gmail note: use an App Password, not your normal password.
    Google Account → Security → "App passwords" → generate one.
"""

import os
import smtplib
import threading
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Module-level deduplication (one email per event per process lifetime) ──────
# This prevents spam if the app retries and hits the same error repeatedly.
_SENT_EVENTS: set[str] = set()
_LOCK = threading.Lock()


def _mark_sent(event_key: str) -> bool:
    """Return True (and mark sent) only if this event_key hasn't been sent yet."""
    with _LOCK:
        if event_key in _SENT_EVENTS:
            return False
        _SENT_EVENTS.add(event_key)
        return True


def send_alert(subject: str, body: str, event_key: str | None = None) -> bool:
    """
    Send an alert email using SMTP credentials from .env.

    Parameters
    ----------
    subject : str
        Email subject line.
    body : str
        Plain-text alert body.
    event_key : str | None
        Deduplication key — the same key will not trigger a second email
        within the same process lifetime.  Pass None to always send.

    Returns
    -------
    bool
        True if a send was dispatched, False if skipped or misconfigured.
    """
    if event_key is not None and not _mark_sent(event_key):
        logger.debug("Alert '%s' already sent this session — skipping duplicate.", event_key)
        return False

    from_addr = os.getenv("ALERT_EMAIL_FROM", "").strip()
    to_addr   = os.getenv("ALERT_EMAIL_TO",   from_addr).strip()
    password  = os.getenv("ALERT_EMAIL_PASSWORD", "").strip()

    if not from_addr or not password:
        logger.warning(
            "Email alert not sent — set ALERT_EMAIL_FROM and ALERT_EMAIL_PASSWORD in .env"
        )
        return False

    host = os.getenv("ALERT_SMTP_HOST", "smtp.gmail.com").strip()
    port = int(os.getenv("ALERT_SMTP_PORT", "587"))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_body = f"""
<html>
<body style="font-family:Arial,sans-serif;background:#f0f2f5;padding:32px;margin:0;">
  <div style="max-width:560px;margin:0 auto;background:#ffffff;border-radius:14px;
              box-shadow:0 4px 20px rgba(0,0,0,0.10);overflow:hidden;">
    <div style="background:#030810;padding:22px 28px;display:flex;align-items:center;gap:12px;">
      <span style="font-size:24px;">✈️</span>
      <span style="color:#ffcf34;font-size:16px;font-weight:700;letter-spacing:-0.02em;">
        Aviation Intelligence Platform
      </span>
    </div>
    <div style="padding:28px 32px;">
      <h2 style="margin:0 0 14px;color:#1a1a2e;font-size:19px;font-weight:700;">
        {subject}
      </h2>
      <div style="background:#f8f9fb;border-left:4px solid #ffcf34;padding:14px 16px;
                  border-radius:0 8px 8px 0;font-size:13px;line-height:1.65;
                  color:#333;white-space:pre-wrap;font-family:monospace;">{body}</div>
      <p style="margin-top:20px;padding-top:16px;border-top:1px solid #eee;
                 color:#aaa;font-size:11px;">
        Sent automatically at {timestamp} &middot; Aviation Intelligence Platform
      </p>
    </div>
  </div>
</body>
</html>
"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[AV-Intel] {subject}"
    msg["From"]    = from_addr
    msg["To"]      = to_addr
    msg.attach(MIMEText(f"[AV-Intel] {subject}\n\n{body}\n\nSent at {timestamp}", "plain"))
    msg.attach(MIMEText(html_body, "html"))

    def _send() -> None:
        try:
            with smtplib.SMTP(host, port, timeout=15) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
                smtp.login(from_addr, password)
                smtp.sendmail(from_addr, [to_addr], msg.as_string())
            logger.info("Alert email sent to %s", to_addr)
        except smtplib.SMTPAuthenticationError:
            logger.warning(
                "Alert email failed — authentication error. "
                "Check ALERT_EMAIL_PASSWORD (use a Gmail App Password, not your normal password)."
            )
        except smtplib.SMTPConnectError:
            logger.warning("Alert email failed — could not connect to %s:%s", host, port)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Alert email failed: %s", exc)

    # Fire-and-forget in a daemon thread so it never blocks the app
    threading.Thread(target=_send, daemon=True, name="alert-mailer").start()
    return True
