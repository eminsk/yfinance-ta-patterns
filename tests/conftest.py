"""Pytest configuration and global fixtures."""

from __future__ import annotations

import warnings

# Ignore LibreSSL runtime warning from urllib3 on older macOS Python builds
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

try:
    import ssl  # noqa: F401
except (ImportError, ModuleNotFoundError):
    import sys
    from unittest.mock import MagicMock
    mock_ssl = MagicMock()
    mock_ssl.OPENSSL_VERSION_INFO = (3, 0, 0, 0, 0)
    mock_ssl.OPENSSL_VERSION = "OpenSSL 3.0.0"
    mock_ssl.create_default_context.return_value = MagicMock()
    sys.modules["ssl"] = mock_ssl
    sys.modules["_ssl"] = mock_ssl
