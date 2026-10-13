import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from agenthood.notify import (
    build_notifiers, notify_all, OsxNotifier, NtfyNotifier, WebhookNotifier,
    TelegramNotifier, EmailNotifier, SmsNotifier, WhatsAppNotifier,
)

# ── build_notifiers ────────────────────────────────────────────────────────────

def test_build_notifiers_original():
    specs = [
        'osx',
        'ntfy:mytopic',
        'ntfy:mytopic:http://custom.ntfy',
        'webhook:https://webhook.site/123',
        'slack:https://hooks.slack.com/123'
    ]
    notifiers = build_notifiers(specs)
    assert len(notifiers) == 5
    assert isinstance(notifiers[0], OsxNotifier)
    assert isinstance(notifiers[1], NtfyNotifier)
    assert notifiers[1].topic == 'mytopic'
    assert notifiers[1].server == 'https://ntfy.sh'
    assert isinstance(notifiers[2], NtfyNotifier)
    assert notifiers[2].server == 'http://custom.ntfy'
    assert isinstance(notifiers[3], WebhookNotifier)
    assert isinstance(notifiers[4], WebhookNotifier)

def test_build_telegram():
    ns = build_notifiers(['telegram:123TOKEN:456CHATID'])
    assert len(ns) == 1
    assert isinstance(ns[0], TelegramNotifier)
    assert ns[0].bot_token == '123TOKEN'
    assert ns[0].chat_id == '456CHATID'

def test_build_telegram_missing_chat_id():
    ns = build_notifiers(['telegram:tokenonly'])
    assert len(ns) == 0  # should warn and skip

def test_build_email_minimal():
    ns = build_notifiers(['email:user@test.com'])
    assert len(ns) == 1
    assert isinstance(ns[0], EmailNotifier)
    assert ns[0].to_addr == 'user@test.com'
    assert ns[0].smtp_host == 'localhost'

def test_build_email_full():
    ns = build_notifiers(['email:to@x.com:smtp.gmail.com:587:me@x.com:pw'])
    assert len(ns) == 1
    n = ns[0]
    assert n.to_addr == 'to@x.com'
    assert n.smtp_host == 'smtp.gmail.com'
    assert n.smtp_port == 587
    assert n.smtp_user == 'me@x.com'
    assert n.smtp_pass == 'pw'

def test_build_sms_textbelt():
    ns = build_notifiers(['sms:+12125551234'])
    assert len(ns) == 1
    assert isinstance(ns[0], SmsNotifier)
    assert ns[0].phone == '+12125551234'
    assert ns[0].provider == 'textbelt'

def test_build_twilio():
    ns = build_notifiers(['twilio:+12125551234:ACSID:AUTHTOKEN:+14155551234'])
    assert len(ns) == 1
    assert isinstance(ns[0], SmsNotifier)
    assert ns[0].provider == 'twilio'
    assert ns[0].account_sid == 'ACSID'
    assert ns[0].from_number == '+14155551234'

def test_build_twilio_too_few_parts():
    ns = build_notifiers(['twilio:+12125551234:SID'])
    assert len(ns) == 0  # should warn and skip

def test_build_whatsapp():
    ns = build_notifiers(['whatsapp:+12125551234:MYAPIKEY'])
    assert len(ns) == 1
    assert isinstance(ns[0], WhatsAppNotifier)
    assert ns[0].phone == '+12125551234'
    assert ns[0].apikey == 'MYAPIKEY'

def test_build_whatsapp_missing_apikey():
    ns = build_notifiers(['whatsapp:+12125551234'])
    assert len(ns) == 0  # should warn and skip

def test_build_unknown():
    ns = build_notifiers(['carrier_pigeon:bob'])
    assert len(ns) == 0

def test_build_mixed():
    specs = [
        'osx', 'telegram:TOKEN:CHAT', 'sms:+1555',
        'whatsapp:+1555:KEY', 'ntfy:topic',
    ]
    ns = build_notifiers(specs)
    assert len(ns) == 5

# ── Individual notifier .send() ────────────────────────────────────────────────

@pytest.mark.asyncio
@patch('asyncio.create_subprocess_exec', new_callable=AsyncMock)
async def test_osx_notifier(mock_exec):
    mock_proc = AsyncMock()
    mock_exec.return_value = mock_proc
    n = OsxNotifier()
    await n.send("MyTitle", "MyBody")
    mock_exec.assert_called_once()
    mock_proc.wait.assert_awaited_once()

@pytest.mark.asyncio
@patch('aiohttp.ClientSession.post')
async def test_ntfy_notifier(mock_post):
    mock_ctx = AsyncMock()
    mock_ctx.status = 200
    mock_post.return_value.__aenter__.return_value = mock_ctx
    n = NtfyNotifier("topic")
    await n.send("Title", "Body")
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == 'https://ntfy.sh/topic'

@pytest.mark.asyncio
@patch('aiohttp.ClientSession.post')
async def test_webhook_notifier(mock_post):
    mock_ctx = AsyncMock()
    mock_ctx.status = 200
    mock_post.return_value.__aenter__.return_value = mock_ctx
    n = WebhookNotifier("http://abc")
    await n.send("Title", "Body")
    mock_post.assert_called_once()

@pytest.mark.asyncio
@patch('aiohttp.ClientSession.post')
async def test_telegram_notifier(mock_post):
    mock_ctx = AsyncMock()
    mock_ctx.status = 200
    mock_post.return_value.__aenter__.return_value = mock_ctx
    n = TelegramNotifier("MYTOKEN", "12345")
    await n.send("Alert", "AAPL dropped")
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert 'MYTOKEN' in args[0]
    assert kwargs['json']['chat_id'] == '12345'
    assert 'Alert' in kwargs['json']['text']

@pytest.mark.asyncio
@patch('smtplib.SMTP')
async def test_email_notifier(mock_smtp):
    mock_server = MagicMock()
    mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
    mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
    n = EmailNotifier('test@test.com', 'localhost', 25, '', '')
    await n.send("Title", "Body")
    mock_smtp.assert_called_once_with('localhost', 25, timeout=10)

@pytest.mark.asyncio
@patch('aiohttp.ClientSession.post')
async def test_sms_textbelt(mock_post):
    mock_ctx = AsyncMock()
    mock_ctx.status = 200
    mock_post.return_value.__aenter__.return_value = mock_ctx
    n = SmsNotifier('+1555')
    await n.send("Alert", "Body")
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert 'textbelt.com' in args[0]

@pytest.mark.asyncio
@patch('aiohttp.ClientSession.post')
async def test_sms_twilio(mock_post):
    mock_ctx = AsyncMock()
    mock_ctx.status = 201
    mock_post.return_value.__aenter__.return_value = mock_ctx
    n = SmsNotifier('+1555', provider='twilio', account_sid='AC123',
                    auth_token='TOKEN', from_number='+1444')
    await n.send("Alert", "Body")
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert 'twilio.com' in args[0]
    assert kwargs['data']['To'] == '+1555'

@pytest.mark.asyncio
@patch('aiohttp.ClientSession.get')
async def test_whatsapp_notifier(mock_get):
    mock_ctx = AsyncMock()
    mock_ctx.status = 200
    mock_get.return_value.__aenter__.return_value = mock_ctx
    n = WhatsAppNotifier('+1555', 'APIKEY')
    await n.send("Alert", "AAPL dropped")
    mock_get.assert_called_once()
    args, _ = mock_get.call_args
    assert 'callmebot' in args[0]
    assert 'APIKEY' in args[0]

# ── notify_all ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_notify_all_swallows_errors():
    n1 = AsyncMock()
    n2 = AsyncMock()
    n2.send.side_effect = Exception("error")
    n3 = AsyncMock()
    await notify_all([n1, n2, n3], "Title", "Body")
    n1.send.assert_awaited_once_with("Title", "Body")
    n2.send.assert_awaited_once_with("Title", "Body")
    n3.send.assert_awaited_once_with("Title", "Body")

@pytest.mark.asyncio
async def test_notify_all_empty():
    await notify_all([], "Title", "Body")  # should not raise


# -- EmailNotifier TLS handling ------------------------------------------------

@pytest.mark.asyncio
async def test_email_port_587_uses_starttls():
    """Port 587 should use SMTP + starttls()."""
    notifier = EmailNotifier(
        to_addr="test@example.com", smtp_host="smtp.example.com",
        smtp_port=587, smtp_user="user", smtp_pass="pass",
    )
    with patch("agenthood.notify.smtplib.SMTP") as MockSMTP:
        mock_server = MagicMock()
        MockSMTP.return_value.__enter__ = MagicMock(return_value=mock_server)
        MockSMTP.return_value.__exit__ = MagicMock(return_value=False)
        notifier._send_sync("Test", "Body")
        MockSMTP.assert_called_once_with("smtp.example.com", 587, timeout=10)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("user", "pass")


@pytest.mark.asyncio
async def test_email_port_465_uses_smtp_ssl():
    """Port 465 should use SMTP_SSL, NOT starttls()."""
    notifier = EmailNotifier(
        to_addr="test@example.com", smtp_host="smtp.example.com",
        smtp_port=465, smtp_user="user", smtp_pass="pass",
    )
    with patch("agenthood.notify.smtplib.SMTP_SSL") as MockSSL, \
         patch("agenthood.notify.smtplib.SMTP") as MockSMTP:
        mock_server = MagicMock()
        MockSSL.return_value.__enter__ = MagicMock(return_value=mock_server)
        MockSSL.return_value.__exit__ = MagicMock(return_value=False)
        notifier._send_sync("Test", "Body")
        MockSSL.assert_called_once_with("smtp.example.com", 465, timeout=10)
        MockSMTP.assert_not_called()
        mock_server.starttls.assert_not_called()


@pytest.mark.asyncio
async def test_email_port_25_no_tls():
    """Port 25 should use plain SMTP without starttls()."""
    notifier = EmailNotifier(
        to_addr="test@example.com", smtp_host="localhost",
        smtp_port=25,
    )
    with patch("agenthood.notify.smtplib.SMTP") as MockSMTP:
        mock_server = MagicMock()
        MockSMTP.return_value.__enter__ = MagicMock(return_value=mock_server)
        MockSMTP.return_value.__exit__ = MagicMock(return_value=False)
        notifier._send_sync("Test", "Body")
        MockSMTP.assert_called_once_with("localhost", 25, timeout=10)
        mock_server.starttls.assert_not_called()

# -- Webhook SSRF validation ---------------------------------------------------

def test_webhook_ssrf_blocks_localhost():
    n = WebhookNotifier('http://127.0.0.1/x')
    assert n._validate_url() is False

def test_webhook_ssrf_blocks_private():
    n = WebhookNotifier('http://10.0.0.1/x')
    assert n._validate_url() is False

def test_webhook_ssrf_blocks_metadata():
    n = WebhookNotifier('http://169.254.169.254/latest')
    assert n._validate_url() is False

def test_webhook_ssrf_blocks_bad_scheme():
    n = WebhookNotifier('ftp://evil.com/x')
    assert n._validate_url() is False

@patch('socket.gethostbyname')
def test_webhook_ssrf_allows_public(mock_gethost):
    mock_gethost.return_value = '93.184.216.34'  # public IP
    n = WebhookNotifier('http://example.com/x')
    assert n._validate_url() is True

def test_webhook_ssrf_blocks_ipv6_loopback():
    n = WebhookNotifier('http://[::1]/x')
    assert n._validate_url() is False

def test_notifier_type_hints_resolve():
    from typing import get_type_hints
    assert get_type_hints(NtfyNotifier.__init__)
    assert get_type_hints(WebhookNotifier.__init__)

