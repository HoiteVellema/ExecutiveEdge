# Stocktwits Success-Rate Dashboard — Fix Build

This build adds **robust HTTP handling** for Streamlit Cloud:
- Retries with backoff for 429/5xx (`Retry-After` honored)
- Friendly error in the UI (instead of a crash)
- Default headers (`User-Agent`, `Accept`)
- Optional token from Streamlit **Secrets** (`ST_ACCESS_TOKEN`) or sidebar

## Deploy on Streamlit Cloud

1. Create a new GitHub repo with `app.py` and `requirements.txt`.
2. In **Advanced settings → Secrets**, add (optional):
   ```
   ST_ACCESS_TOKEN="your_stocktwits_oauth_token"
   ```
3. Deploy with main file `app.py`.

If you still see errors, reduce **Max posts**, double‑check the username, or try later (rate limits).
