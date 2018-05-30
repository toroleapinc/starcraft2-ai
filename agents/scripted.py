"""Scripted baseline agent."""
import numpy as np

class ScriptedAgent:
    """Moves to brightest point on player_relative layer."""
    def select_action(self, screen):
        player_rel = screen[0] if len(screen.shape) == 3 else screen
        target = np.unravel_index(player_rel.argmax(), player_rel.shape)
        return 0, list(target)
