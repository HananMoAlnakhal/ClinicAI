"""Helpers for FSM handler/test integration."""
from __future__ import annotations

from fsm.ui_actions import UIAction


def keyboard_for_action(action: UIAction):
    """Patient flow is text-only — no reply keyboards."""
    return None
