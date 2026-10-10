"""Пакет ADK-агента «Репетитор».

`adk web tutor_agent` или `adk run tutor_agent` загружает `root_agent` из
[`tutor_agent.agent`](tutor_agent/agent.py).
"""

from .agent import root_agent

__all__ = ["root_agent"]
