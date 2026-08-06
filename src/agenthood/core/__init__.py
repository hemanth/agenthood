"""agenthood core -- the agent, its wiring, and its formatting layers.

Split out of the former monolithic ``agenthood/agent.py``:

  prompts.py     -- system prompts and the analyze prompt template
  response.py    -- normalise whatever agentu.infer() returns into a string
  formatting.py  -- render raw Robinhood JSON as markdown (no LLM call)
  llm.py         -- direct OpenAI-compatible HTTP client (bypasses agentu)
  builder.py     -- build and wire the agentu Agent from config
  agent.py       -- the AgentHood class itself

``agenthood.agent`` remains as a re-export shim for backwards compatibility.
"""
__all__ = ["AgentHood"]


def __getattr__(name: str):
    # Lazy so that importing a leaf module (e.g. agenthood.core.response) does
    # not drag in agentu via core.agent.
    if name == "AgentHood":
        from .agent import AgentHood
        return AgentHood
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
