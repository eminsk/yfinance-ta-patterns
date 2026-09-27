"""Pytest configuration and global fixtures."""

from __future__ import annotations

import warnings

# Ignore LibreSSL runtime warning from urllib3 on older macOS Python builds
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
