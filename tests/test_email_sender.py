"""Tests for tools/email_sender.py — email delivery."""
import pytest
from unittest.mock import patch, MagicMock


class TestMarkdownToHtml:
    def test_converts_heading_and_bullets(self):
        from tools.email_sender import markdown_to_html

        md = "## AI News\n- Headline one\n- Headline two"
        html = markdown_to_html(md)
        assert "<h2>" in html
        assert "<li>" in html

    def test_converts_links(self):
        from tools.email_sender import markdown_to_html

        md = "- Story - [Reuters](https://reuters.com/1)"
        html = markdown_to_html(md)
        assert 'href="https://reuters.com/1"' in html


class TestWrapNewsletter:
    def test_wraps_with_header_and_footer(self):
        from tools.email_sender import wrap_newsletter_html

        body_html = "<h2>Section</h2><ul><li>Story</li></ul>"
        result = wrap_newsletter_html(body_html, "2026-02-13")
        assert "AI News Digest" in result
        assert "2026-02-13" in result
        assert "linear-gradient" in result
        assert body_html in result


class TestSendGmail:
    @patch("tools.email_sender.smtplib.SMTP")
    @patch("tools.email_sender.os.getenv")
    def test_sends_email(self, mock_getenv, mock_smtp_cls):
        from tools.email_sender import send_gmail

        mock_getenv.side_effect = lambda k: {
            "GMAIL_USER": "test@gmail.com",
            "GMAIL_PASSWORD": "app-password",
        }.get(k)

        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        send_gmail("Test Subject", "<p>Hello</p>")

        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("test@gmail.com", "app-password")
        mock_server.sendmail.assert_called_once()

    @patch("tools.email_sender.os.getenv")
    def test_raises_without_credentials(self, mock_getenv):
        from tools.email_sender import send_gmail

        mock_getenv.return_value = None

        with pytest.raises(ValueError, match="GMAIL_USER"):
            send_gmail("Subject", "<p>Body</p>")
