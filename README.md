# 📊 Bybit Premiums Dashboard

A powerful Streamlit dashboard for analyzing Bybit perpetual contracts with APR, funding history, leverage data, and more. It includes DNS bypass logic to handle IP-blocking issues.

This project is an attempt to recreate [Liquidity Goblin's](https://x.com/liquiditygoblin/status/1665674397380902912). 

---

## 🚀 Features

- ✅ Bypass DNS or IP blocks with direct Bybit IP fallback
- ✅ Real-time funding, price change, spread, and volume metrics
- ✅ Historical funding rate APR (24h, 3d, 7d, 30d)
- ✅ Maximum leverage extraction per symbol
- ✅ Interactive filters and CSV export
- ✅ Debug mode with detailed metric trace

---

## 💻 Local Setup

### 1. Clone this repo or download the files

```bash
git clone https://github.com/yourusername/bybit-dashboard.git
cd bybit-dashboard
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API credentials

Create a `.env` file in the project root:

```env
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_secret_key_here
```

Or, if using Streamlit secrets (optional), create `.streamlit/secrets.toml`:

```toml
BYBIT_API_KEY = "your_api_key_here"
BYBIT_API_SECRET = "your_secret_key_here"
```

---

## ▶️ Run the App

```bash
streamlit run bybit-premiums.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🧠 Notes

- The app uses IP fallback if Bybit's primary domains are blocked
- Funding rate history is fetched via `/v5/market/funding/history`
- DNS bypass is achieved by modifying the `Host` header while connecting via direct IPs

---

## 📂 File Structure

```
bybit-dashboard/
├── bybit-premiums.py              # Main Streamlit dashboard
├── requirements.txt               # Python dependencies
├── .env                           # (Optional) Local secrets
├── .streamlit/secrets.toml        # (Optional) Streamlit Cloud secrets
├── README.md                      # Project instructions
```

---

## 🔐 Security

- Do not commit `.env` or `secrets.toml` to GitHub
- Add a `.gitignore` like:

```txt
.env
.streamlit/secrets.toml
__pycache__/
```

---

## 📬 Contact
X/Twitter: [TheWarley](https://x.com/horlar_warley).


Maintained by [TheWarley1](https://github.com/TheWarley1). Feel free to fork, contribute, or report issues.
