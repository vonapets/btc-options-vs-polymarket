# BTC Options vs Polymarket — Live Dashboard

Compares options-implied probability distributions (from Deribit) against Polymarket BTC price bracket markets to identify mispricings in real time.

## How it works

1. **Deribit** (public API, no auth needed): fetches BTC call options across all expirations, extracts the IV smile
2. **Black-Scholes N(d₂)**: derives risk-neutral bracket probabilities from the IV smile — `P(lo < BTC < hi) = N(d₂(lo)) − N(d₂(hi))`
3. **Polymarket** (public API): fetches BTC price bracket events by slug (e.g. `bitcoin-price-on-march-17`), reads YES prices
4. **Edge**: `Options Prob − Polymarket YES%` — positive = Polymarket underpriced vs options, negative = overpriced

## Files

| File | Description |
|------|-------------|
| `btc_dashboard.py` | ⭐ Main live dashboard — run this |
| `btc_options_vs_polymarket.py` | Standalone CLI analysis script |
| `public_mcp_server.py` | MCP server wrapping the Public.com SDK (equity options) |
| `_debug_deribit.py` | Debug script: inspect Deribit API response structure |
| `_debug_poly_event.py` | Debug script: inspect Polymarket event/market structure |
| `_debug_polymarket.py` | Debug script: generic Polymarket market search |
| `_inspect_public_sdk.py` | Debug script: introspect Public.com SDK models |
| `_test_public_server.py` | Smoke test for the Public.com MCP server |

## Running the dashboard

```bash
# Install dependencies (one time)
pip install requests

# Run
python3 btc_dashboard.py
```

Opens `http://localhost:8765` automatically. Auto-refreshes every 45 seconds.

## Public.com MCP server (optional)

If you want to use the `public_mcp_server.py` with Claude Desktop, set your API key as an environment variable — **never hardcode it**:

```bash
export PUBLIC_API_KEY=<your_key>
python3 public_mcp_server.py
```

Or configure it in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "publicdotcom": {
      "command": "python3",
      "args": ["/path/to/public_mcp_server.py"],
      "env": { "PUBLIC_API_KEY": "<your_key>" }
    }
  }
}
```

> **Note:** Public.com does not support crypto options (only equity/index). The dashboard uses Deribit for crypto options data.

## Dependencies

- `requests` — HTTP client
- `publicdotcom-py` / `public_api_sdk` — Public.com SDK (for `public_mcp_server.py` only)
- `mcp` / `fastmcp` — MCP server framework (for `public_mcp_server.py` only)
- Python 3.10+

## Security notes

- No API keys are hardcoded anywhere in this codebase
- The dashboard uses only **public, unauthenticated** APIs (Deribit + Polymarket)
- The Public.com API key is read exclusively from the `PUBLIC_API_KEY` environment variable
- Do **not** commit `claude_desktop_config.json` — it contains your API key in the `env` block
