"""Thin passthroughs to the underlying agentu Agent.

These add nothing but lazy-start and config defaults, so they live apart from
the AgentHood logic proper.  Mixed into AgentHood via :class:`AgentuFacade`.
"""
from __future__ import annotations

from typing import Any


class AgentuFacade:
    """Passthrough surface: memory, direct tool calls, cost, loops, serve, eval.

    Expects the host class to provide ``_agent``, ``_cost_tracker``, ``config``
    and an async ``start()``.
    """

    # -- Memory ----------------------------------------------------------------

    def remember(self, content: str, memory_type: str = "fact", importance: float = 0.7) -> None:
        """Store a fact or observation in the agent's long-term memory."""
        if self._agent:
            self._agent.remember(
                content, memory_type=memory_type,
                importance=importance, store_long_term=True,
            )

    def recall(self, query: str, limit: int = 5) -> list:
        """Retrieve memories matching a query."""
        if self._agent is None:
            return []
        return self._agent.recall(query, limit=limit)

    def memory_stats(self) -> dict:
        """Return memory system statistics."""
        if self._agent is None:
            return {}
        return self._agent.get_memory_stats()

    # -- Direct tool access ----------------------------------------------------

    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Directly invoke a Robinhood MCP tool by name (bypasses LLM)."""
        if self._agent is None:
            await self.start()
        return await self._agent.call(tool_name, params)

    # -- Cost tracking ---------------------------------------------------------

    def cost(self) -> dict:
        """Return API usage/cost stats for this session."""
        if self._cost_tracker is None:
            return {"error": "CostTracker not available"}
        return self._cost_tracker.get_usage()

    # -- Loop engineering ------------------------------------------------------

    async def ralph(
        self,
        prompt_file: str,
        max_iterations: int | None = None,
        timeout_minutes: int | None = None,
        on_iteration=None,
    ) -> dict:
        """Run the agent in Ralph autonomous loop mode.

        Ralph continuously reads a PROMPT.md file with checkboxes and
        loops until all are checked off or limits are reached.

        Example PROMPT.md::
            # Morning Scan
            - [ ] Check portfolio for positions down >5%
            - [ ] Flag any RSI > 75 as overbought
            - [ ] Summarise findings
        """
        if self._agent is None:
            await self.start()
        cfg = self.config
        return await self._agent.ralph(
            prompt_file=prompt_file,
            max_iterations=max_iterations or cfg.ralph_max_iterations,
            timeout_minutes=timeout_minutes or cfg.ralph_timeout_minutes,
            on_iteration=on_iteration,
        )

    async def delegate(self, task: str, max_corrections: int = 1, judges: int = 1) -> dict:
        """Delegate a task using the maker-checker pattern.

        A maker sub-agent executes the task; a checker reviews it.
        With judges>1 a panel votes -- use for high-stakes order decisions.
        """
        if self._agent is None:
            await self.start()
        return await self._agent.delegate(task, max_corrections=max_corrections, judges=judges)

    async def best_of(self, n: int, task: str, judge=None) -> dict:
        """Run N parallel agent instances on the same task; judge picks the best.

        Useful for high-confidence analysis:
            result = await agent.best_of(3, "Should I buy NVDA here?")
        """
        if self._agent is None:
            await self.start()
        return await self._agent.best_of(n=n, task=task, judge=judge)

    def findings(self, status: str = "pending", limit: int = 50) -> list[dict]:
        """Get findings from scheduled autonomous scans."""
        if self._agent is None:
            return []
        return self._agent.findings(status=status, limit=limit)

    # -- REST API server -------------------------------------------------------

    async def serve(self, host: str = "0.0.0.0", port: int = 8000, **kwargs) -> None:
        """Start the AgentHood REST + WebSocket API server.

        Endpoints:
            GET  /            -> agent info
            GET  /health      -> health check
            GET  /tools       -> list tools
            POST /process     -> infer (natural language)
            GET  /process/stream -> SSE streaming
            WS   /ws          -> WebSocket
            POST /tasks       -> async background tasks
            GET  /tasks/{id}  -> poll task status

        Example::
            await agent.serve(host="0.0.0.0", port=8000)
        """
        if self._agent is None:
            await self.start()
        from agentu.runtime.serve import serve as _serve
        await _serve(self._agent, host=host, port=port, **kwargs)

    # -- Evaluation ------------------------------------------------------------

    async def evaluate(self, test_cases: list[dict], use_llm_judge: bool = False) -> dict:
        """Run a batch evaluation against predefined test cases.

        Each test case: {"ask": str, "expect": str, "expect_tool": str (optional)}

        Example::
            results = await agent.evaluate([
                {"ask": "price of AAPL", "expect_tool": "get_equity_quotes"},
                {"ask": "my portfolio value", "expect_tool": "get_portfolio"},
            ])
            print(f"Accuracy: {results['accuracy']:.0%}")
        """
        if self._agent is None:
            await self.start()
        from agentu.eval.eval import evaluate as _eval
        result = await _eval(self._agent, test_cases, use_llm_judge=use_llm_judge)
        return result.to_dict() if hasattr(result, "to_dict") else result
