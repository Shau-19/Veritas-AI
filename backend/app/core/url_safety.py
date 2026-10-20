"""
SSRF guard — validates that a user-supplied URL points to a public host
before the backend fetches it (web scraping, MCP server connections).
"""
import ipaddress
import socket
from urllib.parse import urlparse

from fastapi import HTTPException

_BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal"}


def assert_public_url(url: str) -> None:
    """Raise HTTPException(400) if the URL is not a safe, public HTTP(S) URL."""
    if not url.startswith("http://") and not url.startswith("https://"):
        raise HTTPException(400, f"URL must start with http:// or https://: {url}")

    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(400, f"Invalid URL: {url}")

    if hostname.lower() in _BLOCKED_HOSTNAMES:
        raise HTTPException(400, f"URL host is not allowed: {hostname}")

    try:
        resolved_ips = {info[4][0] for info in socket.getaddrinfo(hostname, None)}
    except socket.gaierror:
        raise HTTPException(400, f"Could not resolve host: {hostname}")

    for ip_str in resolved_ips:
        ip = ipaddress.ip_address(ip_str)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise HTTPException(400, f"URL resolves to a disallowed network address: {hostname}")
