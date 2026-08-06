"""Heuristic ticker extraction from free text.

Only used on the failure path: when the LLM could not pick a tool, we make one
last attempt to recognise a ticker in the prompt so a bare "PYPL" still answers.
This is deliberately NOT part of the normal dispatch -- per AGENTS.md, intent
parsing belongs to the LLM.
"""
from __future__ import annotations

import re

_TICKER_RE = re.compile(r'\b([A-Z]{1,5})\b')

# Short English words and trading verbs that would otherwise look like tickers.
STOPWORDS = {
    "AM", "AN", "AS", "AT", "BE", "BY", "DO",
    "IF", "IN", "IS", "ME", "MY", "NO", "OF", "OK",
    "OR", "TO", "UP", "US", "WE", "THE", "AND", "FOR",
    "NOT", "YOU", "CAN", "HAD", "HER", "WAS", "ONE",
    "OUR", "OUT", "HAS", "HIS", "HOW", "ITS", "MAY", "NEW",
    "OLD", "SEE", "WAY", "WHO", "DID", "GET", "HIM",
    "LET", "SAY", "SHE", "TOO", "USE", "BUY", "SELL", "WHAT",
    "SHOW", "PRICE", "STOCK", "QUOTE", "PORTFOLIO", "POSITIONS",
    "SHOULD", "LIKE", "LOOKS",
    # Greetings and question words. Without these, "hi" -> ticker "HI" and the
    # failure path fires a quote lookup for a word the user never meant as one.
    # Deliberately excludes real one-to-three letter tickers such as ALL, ON,
    # SO, NOW, HE, IT -- those stay eligible.
    "I", "HI", "HEY", "YO", "HELLO", "HIYA", "THANK", "SORRY",
    "WHY", "WHEN", "WHERE", "WHICH", "DOES", "WITH", "FROM",
    "THAT", "THIS", "THEY", "THEM", "THEN", "HAVE",
    # Finance jargon users type in caps. None are listed symbols; "AI" (C3.ai)
    # and "ALL" (Allstate) are, so they stay out of this list.
    "ETF", "ETFS", "IPO", "CEO", "CFO", "EPS", "RSI", "MACD", "SMA", "EMA",
    "ATH", "YTD", "NYSE", "GDP", "CPI", "FED", "SEC", "FAQ", "USD",
}


def extract_tickers(text: str, limit: int | None = None) -> list[str]:
    """Return plausible ticker symbols found in *text*, stopwords removed.

    Case is the signal, so it is preserved rather than upper-cased away:

    - A message that is one bare word is a ticker either way -- "nvda", "PYPL".
    - In a longer sentence only ALREADY-uppercase words count. Upper-casing
      first turns "tell me a joke" into the ticker A (Agilent), which is how
      an off-topic question ends up answered with a stock quote.

    Deliberately conservative: this only runs after the LLM failed to route, and
    finding nothing is fine -- the caller falls through to a plain reply.
    """
    stripped = text.strip()
    words = stripped.split()

    # Bare single word: accept any case ("nvda" and "NVDA" both mean the ticker)
    if len(words) == 1:
        cand = words[0].upper().strip("?!.,;:$'\"")
        if _TICKER_RE.fullmatch(cand) and cand not in STOPWORDS:
            return [cand]
        return []

    # Sentence: only words the user actually typed in caps
    found = [t for t in _TICKER_RE.findall(stripped) if t not in STOPWORDS]
    return found[:limit] if limit else found
