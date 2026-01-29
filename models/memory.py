"""
Memory management for VTT server chat sessions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional


def _char_budget_from_tokens(tokens: int) -> int:
    """Rough heuristic: ~4 chars per token."""
    return max(64, tokens * 4)


@dataclass
class MemoryConfig:
    enabled: bool = True
    max_context_tokens: int = 1024


class SessionMemory:
    """
    In-process memory that keeps a rolling window of messages per session (sid).
    
    Storage shape:
      _store[sid] = List[{"role": "user"|"assistant", "content": str}]
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self._store: Dict[str, List[Dict[str, str]]] = {}
        self._lock = asyncio.Lock()
    
    async def get_preamble(self, sid: str) -> Optional[str]:
        """Return conversation history as a text preamble."""
        if not self.config.enabled:
            return None
        
        async with self._lock:
            msgs = list(self._store.get(sid, []))
        
        if not msgs:
            return None
        
        # Build transcript
        transcript = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in msgs
        )
        
        # Trim to budget (keep most recent)
        budget = _char_budget_from_tokens(self.config.max_context_tokens)
        if len(transcript) <= budget:
            return transcript
        
        return transcript[-budget:]
    
    async def add_user_message(self, sid: str, text: str) -> None:
        """Record a user message."""
        if not self.config.enabled or not sid or not text:
            return
        
        entry = {"role": "user", "content": text}
        async with self._lock:
            self._store.setdefault(sid, []).append(entry)
    
    async def add_assistant_message(self, sid: str, text: str) -> None:
        """Record an assistant message."""
        if not self.config.enabled or not sid or not text:
            return
        
        entry = {"role": "assistant", "content": text}
        async with self._lock:
            self._store.setdefault(sid, []).append(entry)
    
    async def clear_session(self, sid: str) -> None:
        """Clear memory for a session."""
        async with self._lock:
            if sid in self._store:
                del self._store[sid]
    
    async def clear_all(self) -> None:
        """Clear all memory."""
        async with self._lock:
            self._store.clear()


def create_memory_from_config(config: dict) -> SessionMemory:
    """
    Create SessionMemory from model config.
    
    Config shape:
    {
      "memory": {
        "enabled": true,
        "max_context_tokens": 1024
      }
    }
    """
    memory_cfg = config.get("memory", {})
    
    return SessionMemory(MemoryConfig(
        enabled=memory_cfg.get("enabled", False),
        max_context_tokens=memory_cfg.get("max_context_tokens", 1024)
    ))
