# Stocktwits Success-Rate Dashboard (Streamlit)

A simple dashboard that pulls a Stocktwits user's messages, parses tickers & direction, extracts text targets/multipliers, computes daily success rates, and (optionally) backtests intraday 1‑minute strategies for recent posts.

> For research/education only. Not investment advice.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy — Streamlit Community Cloud (easiest)

1. Create a new GitHub repo and upload these files (`app.py`, `requirements.txt`).
2. Go to https://share.streamlit.io/ and connect your GitHub, then **New app**.
3. Select your repo, set **Main file** to `app.py`, and **Deploy**.
4. (Optional) Add a **secret** for a Stocktwits OAuth token if you have one.

## Deploy — Hugging Face Spaces

1. Create a new Space → **SDK: Streamlit**.
2. Upload `app.py` and `requirements.txt`.
3. (Optional) Add a secret named `ST_ACCESS_TOKEN` if you want to pass a token via environment.
4. The app will build and run automatically.

## Deploy — Render (alternative)

1. Create a new **Web Service**, connect your GitHub repo.
2. Runtime: Python 3.x.  
3. Build command:
   ```bash
   pip install -r requirements.txt
   ```
4. Start command:
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```

## Configuration

- The dashboard lets you paste an **access token** in the sidebar; you can keep it empty for public reads.
- Intraday backtests use `yfinance` 1‑minute bars (only available for ~7 days). For older intraday, integrate a provider like Polygon.

## Files

- `app.py` – Streamlit dashboard
- `requirements.txt` – Python dependencies
