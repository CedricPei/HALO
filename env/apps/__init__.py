"""Lightweight simulated apps for HALO."""

from .Calendar import Calendar
from .Contacts import Contacts
from .Database import Database
from .Email import Email
from .Files import Files
from .Messaging import Messaging
from .Wallet import Wallet

__all__ = ["Email", "Calendar", "Messaging", "Contacts", "Wallet", "Database", "Files"]
