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

# Canonical company / crypto name -> ticker symbol map.
NAME_TO_TICKER: dict[str, str] = {
    "APPLE": "AAPL", "GOOGLE": "GOOGL", "ALPHABET": "GOOGL",
    "MICROSOFT": "MSFT", "AMAZON": "AMZN", "META": "META",
    "FACEBOOK": "META", "TESLA": "TSLA", "NVIDIA": "NVDA",
    "NETFLIX": "NFLX", "PAYPAL": "PYPL", "AMD": "AMD",
    "INTEL": "INTC", "DISNEY": "DIS", "WALMART": "WMT",
    "JPMORGAN": "JPM", "BERKSHIRE": "BRK.B", "VISA": "V",
    "MASTERCARD": "MA", "SALESFORCE": "CRM", "UBER": "UBER",
    "SPOTIFY": "SPOT", "AIRBNB": "ABNB", "COINBASE": "COIN",
    "ROBINHOOD": "HOOD", "SNAP": "SNAP", "PINTEREST": "PINS",
    "PALANTIR": "PLTR", "SNOWFLAKE": "SNOW", "CROWDSTRIKE": "CRWD",
    "SHOPIFY": "SHOP", "SQUARE": "XYZ", "BLOCK": "XYZ",
    "BROADCOM": "AVGO", "COSTCO": "COST", "ORACLE": "ORCL",
    "BITCOIN": "BTC", "ETHEREUM": "ETH", "SOLANA": "SOL", "DOGECOIN": "DOGE",
    "DOGE": "DOGE", "RIPPLE": "XRP", "CARDANO": "ADA", "AVALANCHE": "AVAX",
    "SHIBA": "SHIB", "POLKADOT": "DOT", "CHAINLINK": "LINK", "LITECOIN": "LTC",
    "UNISWAP": "UNI", "STELLAR": "XLM", "PEPE": "PEPE", "BONK": "BONK",
}

KNOWN_CRYPTO_SYMBOLS: set[str] = {
    "BTC", "ETH", "SOL", "DOGE", "XRP", "ADA", "AVAX", "SHIB", "DOT", "LINK",
    "XDC", "LTC", "BCH", "UNI", "PEPE", "BONK", "SUI", "APT", "NEAR", "ATOM",
    "MATIC", "POL", "ARB", "OP", "ETC", "XLM", "HBAR", "TRUMP",
}


def is_crypto_symbol(sym: str) -> bool:
    """Return True if *sym* (or its base symbol without -USD/USD) is a known crypto asset."""
    s = sym.strip().upper()
    if s.endswith("-USD"):
        s = s[:-4]
    elif len(s) > 3 and s.endswith("USD"):
        s = s[:-3]
    return s in KNOWN_CRYPTO_SYMBOLS

# Unambiguous multi-letter company names safe to match case-insensitively in prose
# (excludes common English words like "apple" only when not in a market query? No,
# in a trading CLI "apple", "nvidia", "microsoft", "paypal", "tesla", "google",
# "amazon", "netflix", "palantir", "coinbase", "bitcoin", "ethereum" are companies).
_PROSE_COMPANY_NAMES = {
    k for k in NAME_TO_TICKER
    if k not in {"VISA", "SQUARE", "BLOCK", "SNAP", "META"}
}

_PRONOUN_RE = re.compile(r"\b(it|its|that|this)\b", re.IGNORECASE)
_MARKET_HINT_RE = re.compile(
    r"\b(price|quote|trading|stock|shares?|buy|sell|hold|analyze|analysis|"
    r"rsi|macd|pe|p/e|fundamentals?|options?|calls?|puts?|chart|worth|value|"
    r"high|low|earnings|compare|vs|doing|look|looking)\b",
    re.IGNORECASE,
)


def extract_tickers(
    text: str,
    limit: int | None = None,
    context_ticker: str | None = None,
) -> list[str]:
    """Return plausible ticker symbols found in *text*, stopwords removed.

    Case is the signal, so it is preserved rather than upper-cased away:

    - A message that is one bare word is a ticker or company name either way --
      "nvda", "NVDA", "nvidia", "apple".
    - In a longer sentence, ALREADY-uppercase words and known company names
      count. Upper-casing everything first would turn "tell me a joke" into the
      ticker A (Agilent).
    - When *context_ticker* is provided and the sentence refers to "it"/"its"
      in a market context with no explicit symbol, *context_ticker* is returned.
    """
    stripped = text.strip()
    if not stripped:
        return []
    words = stripped.split()

    # Bare single word: accept any case ("nvda", "NVDA", "nvidia", "apple")
    if len(words) == 1:
        cand = words[0].upper().strip("?!.,;:$'\"")
        if cand in NAME_TO_TICKER:
            return [NAME_TO_TICKER[cand]]
        if _TICKER_RE.fullmatch(cand) and cand not in STOPWORDS:
            return [cand]
        return []

    # Sentence: 1) words the user actually typed in caps, 2) unambiguous company names
    found: list[str] = []
    for raw_tok in words:
        clean = raw_tok.strip("?!.,;:$'\"()[]{}")
        if not clean:
            continue
        up = clean.upper()
        if clean.isupper() and _TICKER_RE.fullmatch(up) and up not in STOPWORDS:
            sym = NAME_TO_TICKER.get(up, up)
            if sym not in found:
                found.append(sym)
        elif up in _PROSE_COMPANY_NAMES:
            sym = NAME_TO_TICKER[up]
            if sym not in found:
                found.append(sym)
        elif up in KNOWN_CRYPTO_SYMBOLS and up not in {"DOT", "NEAR", "OP", "ATOM", "LINK"} and _MARKET_HINT_RE.search(stripped):
            if up not in found:
                found.append(up)

    if not found and context_ticker and _PRONOUN_RE.search(stripped) and _MARKET_HINT_RE.search(stripped):
        found.append(context_ticker.upper())

    return found[:limit] if limit else found

