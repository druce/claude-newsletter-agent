"""Email delivery for the newsletter agent.

Converts markdown to styled HTML and sends via Gmail SMTP.
Requires GMAIL_USER and GMAIL_PASSWORD environment variables.
"""
from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import markdown as md_lib


def markdown_to_html(newsletter_md: str) -> str:
    """Convert newsletter markdown to HTML."""
    return md_lib.markdown(newsletter_md, extensions=["extra", "codehilite"])


def wrap_newsletter_html(body_html: str, date_str: str) -> str:
    """Wrap newsletter HTML body in styled email template."""
    return f"""\
<div style="max-width: 800px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;">
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 20px; text-align: center; border-radius: 8px 8px 0 0;">
        <h1 style="color: white; margin: 0; font-size: 32px;">AI News Digest</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 16px;">{date_str}</p>
    </div>
    <div style="background: #ffffff; padding: 30px; border-radius: 0 0 8px 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); line-height: 1.6; color: #333;">
        {body_html}
    </div>
    <div style="text-align: center; padding: 20px; color: #666; font-size: 14px;">
        <p>Generated on {date_str} by AI Newsletter Agent</p>
    </div>
</div>"""


def send_gmail(subject: str, html_content: str) -> None:
    """Send an HTML email via Gmail SMTP.

    Requires environment variables:
        GMAIL_USER: Gmail address
        GMAIL_PASSWORD: App-specific password
    """
    email_address = os.getenv("GMAIL_USER")
    password = os.getenv("GMAIL_PASSWORD")

    if not email_address or not password:
        raise ValueError("GMAIL_USER and GMAIL_PASSWORD environment variables required")

    body = f"""\
<html>
    <head></head>
    <body>
    <div>
    {html_content}
    </div>
    </body>
</html>"""

    message = MIMEMultipart()
    message["From"] = email_address
    message["To"] = email_address
    message["Subject"] = subject

    message.attach(MIMEText(body, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(email_address, password)
        server.sendmail(email_address, email_address, message.as_string())
