"""Compatibility shim for root-level client import path.

Canonical client implementation: client/client.py
"""

from client.client import ModGuardClient

__all__ = ["ModGuardClient"]
