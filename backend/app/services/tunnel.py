"""
Tunnel service — manages ngrok tunnel for local dev so that Colab
and other external callers can reach the local backend.

In production set PUBLIC_URL env var to your real domain;
ngrok will not be started.
"""
import os

_tunnel_url: str | None = None


def start_tunnel(port: int = 8000) -> str | None:
    """
    Start an ngrok tunnel if we're in local mode (no PUBLIC_URL set).
    Returns the public HTTPS URL, or None if not applicable.
    """
    global _tunnel_url

    # Production or custom public URL — skip ngrok
    public_url = os.getenv("PUBLIC_URL", "").strip()
    if public_url:
        _tunnel_url = public_url.rstrip("/")
        return _tunnel_url

    try:
        from pyngrok import ngrok, conf

        auth_token = os.getenv("NGROK_AUTH_TOKEN", "")
        if auth_token:
            conf.get_default().auth_token = auth_token

        tunnel = ngrok.connect(port, "http")
        _tunnel_url = tunnel.public_url.rstrip("/")
        print(f"[Tunnel] ngrok active: {_tunnel_url}")
        return _tunnel_url

    except ImportError:
        print("[Tunnel] pyngrok not installed — Colab callbacks will use localhost.")
        _tunnel_url = f"http://127.0.0.1:{port}"
        return _tunnel_url
    except Exception as e:
        print(f"[Tunnel] ngrok failed to start: {e} — falling back to localhost.")
        _tunnel_url = f"http://127.0.0.1:{port}"
        return _tunnel_url


def get_public_url() -> str:
    """Return the current public URL (starts tunnel if not already running)."""
    global _tunnel_url
    if _tunnel_url is None:
        start_tunnel()
    return _tunnel_url or "http://127.0.0.1:8000"


def get_tunnel_status() -> dict:
    url = get_public_url()
    is_ngrok = "ngrok" in url or "ngrok-free" in url
    return {
        "active": True,
        "url": url,
        "is_ngrok": is_ngrok,
        "mode": "ngrok" if is_ngrok else ("custom" if os.getenv("PUBLIC_URL") else "localhost"),
    }
