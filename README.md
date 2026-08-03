# agenthood

Robinhood trading agent -- talk to your portfolio in plain English.

```bash
pip install agenthood
agenthood setup
```

`setup` walks you through Robinhood OAuth and LLM selection. Config saves to `~/.agenthood/config.json`. After that, just run `agenthood`.

## Quick start

```python
import asyncio
from agenthood import AgentHood

async def main():
    async with AgentHood() as agent:
        print(await agent.chat("What are my open positions?"))
        print(await agent.chat("Get me a quote for NVDA and AAPL"))
        print(await agent.chat("Review a limit buy of 5 AAPL at $210"))

asyncio.run(main())
```

`chat()` sends a prompt and returns the response. The agent picks the right Robinhood tools, fetches data, and explains what it found.

## CLI

```bash
agenthood                          # interactive REPL
agenthood "What is TSLA at?"       # one-shot
agenthood --live                   # real orders
agenthood --stream                 # token-by-token
```

Inside the REPL, dot commands skip the LLM entirely:

```
.portfolio          show portfolio (alias: .p)
.positions          show open positions (alias: .pos)
.quote NVDA         quick price quote (alias: .q NVDA)
.analyze PYPL       multi-source analysis
.help               all commands
```

## Direct tool calls

```python
import asyncio, json
from agenthood import AgentHood

async def main():
    async with AgentHood() as agent:
        result = json.loads(await agent.call_tool("robinhood_get_equity_quotes", {
            "symbols": ["NVDA", "AAPL"]
        }))
        print(result["data"]["results"])

asyncio.run(main())
```

`call_tool(name, params)` bypasses the LLM and calls a Robinhood MCP tool directly. 53 tools available.

## Safety

```python
from agenthood import AgentHood, AgentHoodConfig

cfg = AgentHoodConfig(
    dry_run=False,           # default True -- blocks place_* tools
    max_order_value=1000,    # hard cap per order in USD
    allowed_symbols=["AAPL", "NVDA", "TSLA"],
)
```

Safety hooks run at the tool-call level. The LLM cannot talk its way around them.

## Config

All config lives in `~/.agenthood/`:

```
~/.agenthood/
  config.json       # model, api keys, safety settings
  history           # readline history
  sessions/         # saved chat sessions
  watches.json      # price alerts
  triggered.db      # alert history
  logs/             # daemon logs
  daemon.pid        # background process
```

```json
{
  "model": "meta-llama/llama-3.3-70b-instruct:free",
  "api_base": "https://openrouter.ai/api/v1",
  "api_key": "sk-or-...",
  "analyze_model": "llama-3.3-70b-versatile",
  "analyze_api_base": "https://api.groq.com/openai/v1",
  "analyze_api_key": "gsk_...",
  "dry_run": false,
  "max_order_value": 1000,
  "max_turns": 20
}
```

Robinhood token is auto-discovered from `~/.mcp-auth/`. No manual setup needed after the first `npx mcp-remote` auth flow.

## Multi-provider routing

```json
{
  "model": "google/gemma-4-26b-a4b-it:free",
  "model_fallbacks": ["nvidia/nemotron-3-super-120b-a12b:free"],
  "analyze_model": "llama-3.3-70b-versatile",
  "analyze_api_base": "https://api.groq.com/openai/v1"
}
```

`model` handles chat. `analyze_model` handles `.analyze` -- Groq runs analysis in ~1s vs ~14s on free tier. Dot commands use no model at all.

## Autonomous scanning

```python
cfg = AgentHoodConfig(scan_interval=300)  # every 5 minutes
```

The agent wakes up on a schedule, checks for positions down >5%, scans for RSI breakouts, and records findings.

## License

MIT [Hemanth.HM](https://h3manth.com)
