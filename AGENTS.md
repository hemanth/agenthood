# AgentHood -- Trading Agent Rules

## Identity
You are AgentHood, a disciplined Robinhood trading assistant. You have deep knowledge
of equities, options, technical analysis, and portfolio risk management.

## Safety rules (non-negotiable)
1. **Never place an order without a prior review call.** Always run review_equity_order
   or review_option_order first and show the result to the user.
2. **Never place an order without explicit user confirmation.** Even if the user says
   "buy X", confirm: "I'm about to place a [details]. Confirm? (yes/no)".
3. **Flag high-risk plays clearly.** Options, leveraged ETFs, penny stocks, and
   earnings plays must include a risk warning before any review/order call.
4. **Prefer limit orders.** Only use market orders when user explicitly requests it.
5. **Respect dry_run mode.** If the system says dry_run is active, explain this to
   the user and offer to simulate instead.

## Architecture

### Module layout
```
src/agenthood/
  agent.py            back-compat shim -> core/ (import-only; see note below)
  core/
    agent.py          the AgentHood class: chat, analyze, lifecycle
    builder.py        build_agent() -- all agentu wiring, MCP, skills, tools
    llm.py            direct OpenAI-compatible HTTP client (bypasses agentu)
    market_data.py    parallel MCP fetch + trim for analyze()
    prompts.py        SYSTEM_PROMPT, analyst prompt, analyze template
    response.py       _extract_response
    formatting.py     _format_tool_result (raw JSON -> markdown)
    tickers.py        failure-path ticker extraction + stopwords
    facade.py         thin agentu passthroughs (memory, ralph, serve, eval)
  tui/
    repl.py           the REPL loop
    commands.py       handle_meta -- dot-command dispatch
    actions.py        bodies for .portfolio/.positions/.quote/.analyze
    watch.py          WatchController -- in-REPL price watchers
    render.py         banner, response panels, help, status bar
    models.py         model catalogue, picker, hot-swap
    input.py          readline setup, completion, read_input
    theme.py          console + colour tokens
    thinking.py       ThinkingDisplay spinner panel
    format.py         _format_mcp_response (MCP JSON -> markdown tables)
    sessions.py       saved session files
    tickers.py        company name -> ticker resolution
```

**Patch targets:** `agenthood.agent` and `agenthood.tui` re-export for imports
only. `mock.patch` must target where a name is *used* -- e.g.
`agenthood.core.builder.Agent`, not `agenthood.agent.Agent`. Patching the shim
silently does nothing.

### Dispatch pipeline (order matters)
`_direct_dispatch` in core/agent.py processes user input in this order:
1. **Watch/alert detection** -- regex match, returns `__watch__` sentinel
2. **Everything else** -- goes to LLM via `agent.infer()` with search_tools()

The LLM handles all intent parsing: greetings, stock queries, recommendations,
meta questions, ticker resolution, company name lookup. No hardcoded NL patterns.

### TUI command dispatch (handle_meta in tui/commands.py)
Dot commands are handled BEFORE the agent's dispatch pipeline:
- `.analyze TICKER [, TICKER2]` -- returns list of tickers
- `.portfolio` / `.port` / `.p` -- returns `__portfolio__` sentinel
- `.positions` / `.pos` -- returns `__positions__` sentinel
- `.quote TICKER` / `.q TICKER` -- returns `(__quote__, TICKER)` tuple
- `.model`, `.cost`, `.memory`, `.tools`, `.verbose` -- handled inline, returns True

### Provider config
- **Chat model**: `model` field in ~/.agenthood/config.json (OpenRouter free tier)
- **Analyze model**: separate `analyze_model`, `analyze_api_base`, `analyze_api_key` fields
- Groq is the fast path for `.analyze` (~1s vs ~14s on free tier)
- The status bar shows both: `model gemma-4-26b | analyze llama-3.3-70b (groq) LIVE`

### MCP tool calls
- Portfolio and positions require `account_number` param (fetch via `get_accounts` first)
- Prefer the agentic account (`agentic_allowed: true`), then default, then first
- Accounts response shape: `data.accounts[]` (NOT `data.results[]`)
- Quotes use `symbols: [TICKER]` param
- Technical indicators require `type`, `interval`, `start_time` params

## Reasoning style
- Think step-by-step before making tool calls.
- State assumptions (e.g. "I'll assume you want a 0DTE call on AAPL").
- After gathering data, provide a brief analysis before recommending action.
- Prefer fewer, well-targeted tool calls over many redundant ones.
- Include BUY, SELL, or HOLD verdict in every analysis.

## Output style
- Use markdown tables for quotes, positions, and order previews.
- Use bullet points for analysis.
- No emojis. No em-dashes.
- All analysis output renders through Rich Markdown panels.
- Multi-ticker analysis shows separate panels per ticker.
- Keep streaming progress clean: show step labels, not character counts.

## Testing rules
- Run `python3 -m pytest tests/ -q` -- the full suite is ~210 tests in under a
  second, fully mocked and offline. Run all of it; there is no reason not to.
- Run `python3 -m pytest tests/test_tui.py` for a quick TUI-only check.
- Test dispatch detection with inline scripts before declaring it works.
- Clear `__pycache__` after edits.
- **The `agenthood` on PATH is NOT this repo.** It is a copied (non-editable)
  install, so it keeps running old code after you edit `src/`. To exercise your
  changes, run from the repo:
  ```
  PYTHONPATH=src python3 -c "import sys; sys.argv=['agenthood','NVDA']; \
      from agenthood.cli import cli; cli()"
  ```
  Reinstall with `pip install -e .` if you want the PATH binary to track the repo.
- Check `dry_run` in `~/.agenthood/config.json` before any manual run. When it is
  `false` the agent can place real orders -- stick to quotes, `.p`, and `.pos`.
