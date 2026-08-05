from __future__ import annotations
import asyncio, json, logging, smtplib
from abc import ABC, abstractmethod
from email.message import EmailMessage

logger = logging.getLogger(__name__)

__all__ = ["Notifier", "OsxNotifier", "NtfyNotifier", "WebhookNotifier",
           "TelegramNotifier", "EmailNotifier", "SmsNotifier",
           "WhatsAppNotifier", "build_notifiers", "notify_all"]


class Notifier(ABC):
    """Base notifier. Subclasses implement send(title, body)."""
    @abstractmethod
    async def send(self, title: str, body: str) -> None: ...


class OsxNotifier(Notifier):
    """macOS Notification Center via osascript."""
    async def send(self, title: str, body: str) -> None:
        script = (
            f'display notification {json.dumps(body)} '
            f'with title {json.dumps(title)}'
        )
        proc = await asyncio.create_subprocess_exec(
            'osascript', '-e', script,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()


class NtfyNotifier(Notifier):
    """Push notification via ntfy.sh (https://ntfy.sh)."""
    def __init__(self, topic: str, server: str = 'https://ntfy.sh', session: Optional[Any] = None):
        self.topic = topic
        self.server = server.rstrip('/')
        self.session = session

    async def send(self, title: str, body: str) -> None:
        import aiohttp
        url = f'{self.server}/{self.topic}'
        headers = {'Title': title, 'Priority': 'high',
                   'Tags': 'chart_with_upwards_trend'}
        sess = self.session
        close_sess = False
        if sess is None:
            sess = aiohttp.ClientSession()
            close_sess = True
        try:
            async with sess.post(url, data=body.encode(), headers=headers) as r:
                if r.status not in (200, 201):
                    logger.warning('ntfy returned %s', r.status)
        finally:
            if close_sess:
                await sess.close()


class WebhookNotifier(Notifier):
    """POST JSON to a webhook URL (Discord, Slack, Teams, custom)."""
    def __init__(self, url: str, session: Optional[Any] = None):
        self.url = url
        self.session = session

    def _validate_url(self) -> bool:
        """Block SSRF -- reject private/loopback/link-local targets."""
        from urllib.parse import urlparse
        import ipaddress
        import socket
        parsed = urlparse(self.url)
        if parsed.scheme not in ('https', 'http'):
            logger.warning('Webhook blocked: unsupported scheme %s', parsed.scheme)
            return False
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(parsed.hostname or ''))
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                logger.warning('Webhook blocked: private/loopback target')
                return False
        except (socket.gaierror, ValueError):
            pass  # unresolvable host -- let aiohttp handle the error
        return True

    async def send(self, title: str, body: str) -> None:
        if not self._validate_url():
            return
        import aiohttp
        payload = {
            'text': f'**{title}**\n{body}',
            'content': f'**{title}**\n{body}',
            'title': title, 'body': body,
        }
        sess = self.session
        close_sess = False
        if sess is None:
            sess = aiohttp.ClientSession()
            close_sess = True
        try:
            async with sess.post(self.url, json=payload) as r:
                if r.status not in (200, 201, 204):
                    from urllib.parse import urlparse
                    parsed = urlparse(self.url)
                    safe_url = f"{parsed.scheme}://{parsed.netloc}/..."
                    logger.warning('Webhook %s returned %s', safe_url, r.status)
        finally:
            if close_sess:
                await sess.close()


class TelegramNotifier(Notifier):
    """Telegram via Bot API. Free, instant, works on any device.

    Setup:
      1. Message @BotFather on Telegram, /newbot, get the bot token
      2. Start a chat with your bot, send /start
      3. Get your chat_id from https://api.telegram.org/bot<TOKEN>/getUpdates

    Spec: ``telegram:BOT_TOKEN:CHAT_ID``
    """
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id

    async def send(self, title: str, body: str) -> None:
        import aiohttp
        url = f'https://api.telegram.org/bot{self.bot_token}/sendMessage'
        payload = {
            'chat_id': self.chat_id,
            'text': f'*{title}*\n{body}',
            'parse_mode': 'Markdown',
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as r:
                if r.status != 200:
                    logger.warning('Telegram returned %s', r.status)


class EmailNotifier(Notifier):
    """Email via SMTP. Works with Gmail (app password), any SMTP server.

    Spec: ``email:to@addr:smtp_host:port:user:password``
    Minimal: ``email:to@addr`` (uses localhost:25, no auth)
    """
    def __init__(self, to_addr: str, smtp_host: str = 'localhost',
                 smtp_port: int = 587, smtp_user: str = '',
                 smtp_pass: str = '', from_addr: str = ''):
        self.to_addr = to_addr
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_pass = smtp_pass
        self.from_addr = from_addr or smtp_user or 'agenthood@localhost'

    async def send(self, title: str, body: str) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._send_sync, title, body)

    def _send_sync(self, title: str, body: str) -> None:
        msg = EmailMessage()
        msg['Subject'] = title
        msg['From'] = self.from_addr
        msg['To'] = self.to_addr
        msg.set_content(body)
        try:
            if self.smtp_port == 465:
                # Port 465 uses implicit SSL (SMTPS)
                with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port,
                                      timeout=10) as s:
                    if self.smtp_user:
                        s.login(self.smtp_user, self.smtp_pass)
                    s.send_message(msg)
            else:
                with smtplib.SMTP(self.smtp_host, self.smtp_port,
                                  timeout=10) as s:
                    if self.smtp_port not in (25,):
                        s.starttls()
                    if self.smtp_user:
                        s.login(self.smtp_user, self.smtp_pass)
                    s.send_message(msg)
        except Exception as e:
            logger.warning('Email send failed: %s', e)


class SmsNotifier(Notifier):
    """SMS via TextBelt (free: 1/day to own number) or Twilio.

    TextBelt spec: ``sms:+1234567890``  (free, 1/day)
    Twilio spec:   ``twilio:+1234567890:ACCOUNT_SID:AUTH_TOKEN:+1FROM``
    """
    def __init__(self, phone: str, provider: str = 'textbelt',
                 account_sid: str = '', auth_token: str = '',
                 from_number: str = ''):
        self.phone = phone
        self.provider = provider
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number

    async def send(self, title: str, body: str) -> None:
        import aiohttp
        text = f'{title}\n{body}'

        if self.provider == 'twilio' and self.account_sid:
            url = (f'https://api.twilio.com/2010-04-01/Accounts/'
                   f'{self.account_sid}/Messages.json')
            data = {
                'To': self.phone,
                'From': self.from_number,
                'Body': text,
            }
            auth = aiohttp.BasicAuth(self.account_sid, self.auth_token)
            async with aiohttp.ClientSession(auth=auth) as session:
                async with session.post(url, data=data) as r:
                    if r.status not in (200, 201):
                        logger.warning('Twilio returned %s', r.status)
        else:
            # TextBelt free tier (1 SMS/day to own number)
            url = 'https://textbelt.com/text'
            payload = {'phone': self.phone, 'message': text,
                       'key': 'textbelt'}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as r:
                    if r.status != 200:
                        logger.warning('TextBelt returned %s', r.status)


class WhatsAppNotifier(Notifier):
    """WhatsApp via CallMeBot (free, unofficial).

    Setup:
      1. Add +34 644 51 95 23 to your contacts
      2. Send "I allow callmebot to send me messages" via WhatsApp
      3. You'll receive an apikey

    Spec: ``whatsapp:+1234567890:APIKEY``
    """
    def __init__(self, phone: str, apikey: str):
        self.phone = phone
        self.apikey = apikey

    async def send(self, title: str, body: str) -> None:
        import aiohttp
        from urllib.parse import quote_plus
        text = f'*{title}*\n{body}'
        url = (f'https://api.callmebot.com/whatsapp.php'
               f'?phone={quote_plus(self.phone)}'
               f'&text={quote_plus(text)}'
               f'&apikey={self.apikey}')
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as r:
                if r.status != 200:
                    logger.warning('CallMeBot returned %s', r.status)


# ── Builder ────────────────────────────────────────────────────────────────────

def build_notifiers(notify_specs: list[str]) -> list[Notifier]:
    """Parse notifier specs into Notifier instances.

    Supported specs::

        osx
        ntfy:topic
        ntfy:topic:server_url
        webhook:url  |  slack:url  |  discord:url
        telegram:BOT_TOKEN:CHAT_ID
        email:to@addr:smtp_host:port:user:password
        sms:+1234567890                                 (TextBelt free)
        twilio:+1234567890:ACCOUNT_SID:AUTH_TOKEN:+FROM
        whatsapp:+1234567890:APIKEY                     (CallMeBot free)
    """
    notifiers: list[Notifier] = []
    for spec in notify_specs:
        try:
            if spec == 'osx':
                notifiers.append(OsxNotifier())
            elif spec.startswith('ntfy:'):
                parts = spec.split(':', 2)
                topic = parts[1]
                server = parts[2] if len(parts) > 2 else 'https://ntfy.sh'
                notifiers.append(NtfyNotifier(topic, server))
            elif spec.startswith(('webhook:', 'slack:', 'discord:')):
                url = spec.split(':', 1)[1]
                notifiers.append(WebhookNotifier(url))
            elif spec.startswith('telegram:'):
                parts = spec.split(':', 2)
                if len(parts) < 3:
                    logger.warning('telegram spec needs BOT_TOKEN:CHAT_ID')
                    continue
                notifiers.append(TelegramNotifier(parts[1], parts[2]))
            elif spec.startswith('email:'):
                parts = spec.split(':')
                to_addr = parts[1]
                host = parts[2] if len(parts) > 2 else 'localhost'
                port = int(parts[3]) if len(parts) > 3 else 587
                user = parts[4] if len(parts) > 4 else ''
                pwd = parts[5] if len(parts) > 5 else ''
                notifiers.append(EmailNotifier(to_addr, host, port,
                                               user, pwd))
            elif spec.startswith('sms:'):
                phone = spec.split(':', 1)[1]
                notifiers.append(SmsNotifier(phone))
            elif spec.startswith('twilio:'):
                parts = spec.split(':')
                if len(parts) < 5:
                    logger.warning(
                        'twilio spec needs +phone:SID:TOKEN:+FROM')
                    continue
                notifiers.append(SmsNotifier(
                    parts[1], provider='twilio',
                    account_sid=parts[2], auth_token=parts[3],
                    from_number=parts[4],
                ))
            elif spec.startswith('whatsapp:'):
                parts = spec.split(':', 2)
                if len(parts) < 3:
                    logger.warning('whatsapp spec needs +phone:APIKEY')
                    continue
                notifiers.append(WhatsAppNotifier(parts[1], parts[2]))
            else:
                logger.warning('Unknown notifier spec: %r', spec)
        except Exception as e:
            logger.warning('Failed to build notifier from %r: %s', spec, e)
    return notifiers


async def notify_all(notifiers: list[Notifier], title: str,
                     body: str) -> None:
    if not notifiers:
        return
    results = await asyncio.gather(
        *[n.send(title, body) for n in notifiers],
        return_exceptions=True,
    )
    for r in results:
        if isinstance(r, Exception):
            logger.warning('Notifier error: %s', r)

