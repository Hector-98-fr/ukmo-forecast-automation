import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_email(html_table: str):

    sender = os.environ["EMAIL_SENDER"]
    receiver = os.environ["EMAIL_RECEIVER"]
    password = os.environ["EMAIL_PASSWORD"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "UKMO Forecast (Automated)"
    msg["From"] = sender
    msg["To"] = receiver

    html = f"""
    <html>
      <body>
        <h2>UKMO Forecast</h2>
        {html_table}
      </body>
    </html>
    """

    msg.attach(MIMEText(html, "html"))

    # Outlook / Live SMTP server
    with smtplib.SMTP("smtp-mail.outlook.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
