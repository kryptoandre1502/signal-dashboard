# signal_dashboard_full_complete.py
# Vollständiges Streamlit Dashboard mit:
# - Watchlist (Crypto / Forex / Aktien)
# - Suchfeld zum Hinzufügen von Symbolen
# - KI-Signal Engine (mit erweiterten Indikatoren: HalfTrend(4h), RSI(1h), ATR(1h), MFI)
# - Telegram (Human + JSON) Versand
# - Candlestick-Charts mit Kalman-Filter, SL/TP und Trade-Markern
# - Backtest mit Equity, Buy&Hold, Trades, Max Drawdown (Equity) und Max Trade Drawdown (MAE)
#
# Installation (falls nötig):
# pip install streamlit yfinance pandas_ta scikit-learn pykalman plotly requests

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from pykalman import KalmanFilter
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
import json
import warnings
import logging

warnings.filterwarnings('ignore')

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("SignalDashboard")

# -------------------------
# Page config + Theme CSS
# -------------------------
PRIMARY = "#0B6E4F"
ACCENT  = "#1F77B4"
TEXT    = "#F7FAFC"

st.set_page_config(page_title="Signal Dashboard + Backtest", layout="wide")
st.markdown(f"""
<style>
:root {{ --primary: {PRIMARY}; --accent: {ACCENT}; --text: {TEXT}; }}
html, body, .stApp {{ background: linear-gradient(180deg,#071019 0%,#071a1f 100%); color: var(--text); }}
.stButton>button {{ background: linear-gradient(90deg,var(--primary),var(--accent)); color: white; }}
.card {{ background: rgba(255,255,255,0.03); border-radius:10px; padding:10px; margin-bottom:8px; border:1px solid rgba(255,255,255,0.04); }}
.small-muted {{ color: rgba(255,255,255,0.6); font-size:12px; }}
.signal-buy {{ color: #00ff88; font-weight:700; }}
.signal-sell {{ color: #ff6b6b; font-weight:700; }}
.signal-neutral {{ color: #cccccc; font-weight:700; }}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Defaults / Examples
# -------------------------
COINS_FEST = {
    "BTC-USD": {"yf":"BTC-USD","name":"Bitcoin", "farbe":"#F7931A","typ":"Krypto"},
    "ETH-USD": {"yf":"ETH-USD","name":"Ethereum","farbe":"#627EEA","typ":"Krypto"},
    "BNB-USD": {"yf":"BNB-USD","name":"BNB",     "farbe":"#F3BA2F","typ":"Krypto"},
}
EXAMPLES_CRYPTO = ["BTC-USD","ETH-USD","SOL-USD","XRP-USD","ADA-USD","DOGE-USD","AVAX-USD"]
EXAMPLES_FOREX  = ["EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X"]
EXAMPLES_STOCKS = ["AAPL","TSLA","MSFT","NVDA","AMZN","GOOGL","META"]
FARBEN_EXTRA = ["#E74C3C","#8E44AD","#2ECC71","#1ABC9C","#3498DB","#F39C12","#D35400","#C0392B","#16A085","#27AE60"]

# -------------------------
# Helper functions
# -------------------------
def richtungs_prob(prob):
    return f"{prob:.1f}% Aufwaerts" if prob >= 50 else f"{100-prob:.1f}% Abwaerts"

def signal_farbe(sig):
    return {"BUY":"#00ff88","SELL":"#ff6b6b"}.get(sig,"#cccccc")

def signal_label(sig):
    return {"BUY":"BUY / KAUFEN","SELL":"SELL / VERKAUFEN"}.get(sig,"NEUTRAL / ABWARTEN")

def berechne_sl_tp(preis, sig, atr, sl_mult, tp_mult):
    if preis is None or preis == 0:
        return None, None, None, None
    if not atr or np.isnan(float(atr)) or float(atr) <= 0:
        atr = preis * 0.02
    atr = float(atr)
    if sig == "BUY":
        sl, tp = preis - sl_mult*atr, preis + tp_mult*atr
    elif sig == "SELL":
        sl, tp = preis + sl_mult*atr, preis - tp_mult*atr
    else:
        return None, None, None, None
    return sl, tp, (sl-preis)/preis*100, (tp-preis)/preis*100

def yf_sym(eingabe, typ):
    s = eingabe.strip().upper()
    if typ == "Krypto" and "-USD" not in s and "USDT" not in s:
        s = s.replace("USDT","") + "-USD"
    return s

# -------------------------
# Telegram helpers (human + JSON)
# -------------------------
def sende_telegram(token, chat_id, text):
    if not token or not chat_id:
        return False, "Token/Chat-ID fehlt"
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token.strip()}/sendMessage",
            json={"chat_id":chat_id.strip(),"text":text,"parse_mode":"Markdown"},
            timeout=10
        )
        d = r.json()
        return (True,"Gesendet") if r.status_code==200 else (False,d.get("description","Fehler"))
    except Exception as e:
        logger.exception("Telegram senden fehlgeschlagen")
        return False, str(e)

def tg_nachricht_human(name, sym, sig, preis, prob, konf, zr, treff, sl, tp, sl_pct, tp_pct, typ):
    sl_s = f"${sl:,.4f} ({sl_pct:+.2f}%)" if sl else "-"
    tp_s = f"${tp:,.4f} ({tp_pct:+.2f}%)" if tp else "-"
    return (
        f"*KI SIGNAL - {name} ({sym})*\n"
        f"Typ: {typ} | Zeitrahmen: {zr}\n"
        f"{'-'*30}\n"
        f"Signal:       *{signal_label(sig)}*\n"
        f"Kurs:         ${preis:,.4f}\n"
        f"Richtung:     {richtungs_prob(prob)}\n"
        f"Konfidenz:    {konf}\n"
        f"KI-Genauigk.: {treff:.1f}%\n"
        f"{'-'*30}\n"
        f"Stop-Loss:    {sl_s}\n"
        f"Take-Profit:  {tp_s}\n"
        f"{'-'*30}\n"
        f"{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}"
    )

def tg_nachricht_json(name, sym, sig, preis, prob, konf, zr, treff, sl, tp, sl_pct, tp_pct, typ):
    payload = {
        "type": "TRADE_SIGNAL",
        "symbol": sym,
        "action": sig,
        "price": round(float(preis), 6),
        "stop_loss": round(float(sl), 6) if sl else None,
        "take_profit": round(float(tp), 6) if tp else None,
        "confidence": float(prob),
        "confidence_level": konf,
        "timeframe": zr,
        "ai_accuracy": round(float(treff), 1),
        "asset_name": name,
        "asset_type": typ,
        "timestamp": datetime.now().isoformat()
    }
    return json.dumps(payload, ensure_ascii=False)

def sende_signal_beide(token, chat_id, name, sym, sig, preis, prob, konf, zr, treff, sl, tp, sl_pct, tp_pct, typ):
    if not token or not chat_id:
        return False, "Token/Chat-ID fehlt"
    text_human = tg_nachricht_human(name, sym, sig, preis, prob, konf, zr, treff, sl, tp, sl_pct, tp_pct, typ)
    text_json = tg_nachricht_json(name, sym, sig, preis, prob, konf, zr, treff, sl, tp, sl_pct, tp_pct, typ)
    ok1, err1 = sende_telegram(token, chat_id, text_human)
    ok2, err2 = sende_telegram(token, chat_id, text_json)
    if ok1 and ok2:
        logger.info(f"Telegram: human+json für {sym} gesendet")
        return True, "Beide Nachrichten gesendet"
    else:
        logger.warning(f"Telegram Fehler: {err1 or err2}")
        return False, f"Fehler: {err1 or err2}"

# -------------------------
# Data loader
# -------------------------
@st.cache_data(ttl=600)
def lade_daten(symbol, interval, days):
    try:
        end = datetime.now(); start = end - timedelta(days=days)
        d = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
        if d.empty: return pd.DataFrame()
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        df = d.reset_index()
        if 'Datetime' in df.columns: df = df.rename(columns={'Datetime':'Date'})
        elif 'Date' not in df.columns: df['Date'] = df.index
        return df
    except Exception:
        logger.exception("lade_daten fehlgeschlagen")
        return pd.DataFrame()

def validiere(symbol):
    # Validierung erweitert: prüfe 1d + 1h Verfügbarkeit (mindestens 3 Tage 1d, 60 Kerzen 1h)
    df1d = lade_daten(symbol, "1d", 10)
    if df1d.empty or len(df1d) < 3:
        return False
    df1h = lade_daten(symbol, "1h", 10)
    if df1h.empty or len(df1h) < 60:
        return False
    return True

# -------------------------
# Extra indicators computation (1h/4h/MFI)
# -------------------------
def compute_extra_indicators(symbol):
    """
    Lädt 1h und 4h Daten (falls verfügbar) und berechnet:
      - RSI_1h (14)
      - ATR_1h (14)
      - MFI_1h (14)
      - HalfTrend_4h (wenn möglich) -> fallback: Kalman slope on 4h
    Gibt dict mit Werten oder None.
    """
    out = {"rsi_1h": None, "atr_1h": None, "mfi_1h": None, "halftrend_4h": None}
    try:
        df1h = lade_daten(symbol, "1h", 90)
        if not df1h.empty and len(df1h) >= 20:
            df1h['RSI_1h'] = ta.rsi(df1h['Close'], length=14)
            df1h['ATR_1h'] = ta.atr(df1h['High'], df1h['Low'], df1h['Close'], length=14)
            try:
                df1h['MFI_1h'] = ta.mfi(df1h['High'], df1h['Low'], df1h['Close'], df1h['Volume'], length=14)
            except Exception:
                df1h['MFI_1h'] = np.nan
            out['rsi_1h'] = float(df1h['RSI_1h'].dropna().values[-1]) if 'RSI_1h' in df1h.columns and not df1h['RSI_1h'].dropna().empty else None
            out['atr_1h'] = float(df1h['ATR_1h'].dropna().values[-1]) if 'ATR_1h' in df1h.columns and not df1h['ATR_1h'].dropna().empty else None
            out['mfi_1h'] = float(df1h['MFI_1h'].dropna().values[-1]) if 'MFI_1h' in df1h.columns and not df1h['MFI_1h'].dropna().empty else None
    except Exception:
        logger.exception("compute_extra_indicators: 1h failed")

    # HalfTrend 4h
    try:
        df4h = lade_daten(symbol, "4h", 180)
        if not df4h.empty and len(df4h) >= 30:
            ht_val = None
            try:
                # Try pandas_ta halftrend if available
                if hasattr(ta, "halftrend"):
                    ht = ta.halftrend(df4h['Close'], length=10)
                    if isinstance(ht, (pd.Series, np.ndarray)):
                        ht_val = float(pd.Series(ht).dropna().values[-1])
                    elif isinstance(ht, pd.DataFrame):
                        ht_val = float(ht.iloc[:,0].dropna().values[-1])
                else:
                    # fallback: use Kalman slope on 4h close as proxy for trend sign
                    kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1],
                                      initial_state_mean=float(df4h['Close'].values[0]),
                                      initial_state_covariance=1, observation_covariance=2, transition_covariance=0.05)
                    sm, _ = kf.filter(df4h['Close'].values)
                    slope = pd.Series(np.asarray(sm).ravel()).diff().dropna().tail(3).mean()
                    ht_val = float(slope)
            except Exception:
                logger.exception("HalfTrend calc failed; using fallback")
                ht_val = None
            out['halftrend_4h'] = ht_val
    except Exception:
        logger.exception("compute_extra_indicators: 4h failed")

    return out

# -------------------------
# KI Signal Engine (erweitert um extra_indicators)
# -------------------------
def berechne_signal(df, threshold, kf_filter, extra_indicators=None):
    """
    df: DataFrame for the timeframe being evaluated (1h or 1d)
    extra_indicators: dict with keys 'rsi_1h','atr_1h','mfi_1h','halftrend_4h' (values or None)
    """
    try:
        if len(df) < 60: return 50.0,"NIEDRIG","NEUTRAL",50.0,None
        # Kalman smoothing
        kf = KalmanFilter(transition_matrices=[1],observation_matrices=[1],
            initial_state_mean=float(df['Close'].values[0]),
            initial_state_covariance=1,observation_covariance=2,transition_covariance=0.05)
        sm,_ = kf.filter(df['Close'].values)
        df['Kalman_Price'] = np.asarray(sm).ravel()
        df['Kalman_Slope'] = pd.Series(df['Kalman_Price']).diff().values

        # Standard indicators
        df['RSI']      = ta.rsi(df['Close'],length=14)
        df['RSI_fast'] = ta.rsi(df['Close'],length=7)
        df['RSI_slow'] = ta.rsi(df['Close'],length=21)
        macd = ta.macd(df['Close'],fast=12,slow=26,signal=9)
        if macd is not None and not macd.empty:
            df['MACD']=macd.iloc[:,0]; df['MACD_signal']=macd.iloc[:,1]; df['MACD_hist']=macd.iloc[:,2]
        else:
            df['MACD']=df['MACD_signal']=df['MACD_hist']=0.0
        bb = ta.bbands(df['Close'],length=20)
        if bb is not None and not bb.empty:
            df['BB_width']=(bb.iloc[:,0]-bb.iloc[:,2])/bb.iloc[:,1]
            df['BB_position']=(df['Close']-bb.iloc[:,2])/(bb.iloc[:,0]-bb.iloc[:,2])
        else:
            df['BB_width']=df['BB_position']=0.0
        df['ATR']           = ta.atr(df['High'],df['Low'],df['Close'],length=14)
        df['Log_Return']    = np.log(df['Close']/df['Close'].shift(1))
        df['Volatility_20'] = df['Log_Return'].rolling(20).std()
        df['Momentum_5']    = df['Close']/df['Close'].shift(5)-1
        df['Momentum_10']   = df['Close']/df['Close'].shift(10)-1
        df['Momentum_20']   = df['Close']/df['Close'].shift(20)-1
        df['Volume_Ratio']  = df['Volume']/df['Volume'].rolling(20).mean()
        df['Z_Score_Kalman']  = (df['Close']-df['Kalman_Price'])/df['Close'].rolling(20).std()
        df['Kalman_Slope_sm'] = pd.Series(df['Kalman_Slope']).rolling(5).mean().values
        df['Price_vs_SMA20']  = df['Close']/df['Close'].rolling(20).mean()-1
        df['Price_vs_SMA50']  = df['Close']/df['Close'].rolling(50).mean()-1
        df['Higher_High']     = (df['High']>df['High'].shift(1)).astype(int)
        df['Lower_Low']       = (df['Low']<df['Low'].shift(1)).astype(int)
        df['Target']          = np.where(df['Close'].shift(-1)>df['Close'],1,0)

        # Integrate extra indicators as constant columns (if provided)
        if extra_indicators:
            if extra_indicators.get('rsi_1h') is not None:
                df['RSI_1H'] = float(extra_indicators.get('rsi_1h'))
            else:
                df['RSI_1H'] = np.nan
            if extra_indicators.get('atr_1h') is not None:
                df['ATR_1H'] = float(extra_indicators.get('atr_1h'))
            else:
                df['ATR_1H'] = np.nan
            if extra_indicators.get('mfi_1h') is not None:
                df['MFI_1H'] = float(extra_indicators.get('mfi_1h'))
            else:
                df['MFI_1H'] = np.nan
            if extra_indicators.get('halftrend_4h') is not None:
                df['HALFTREND_4H'] = float(extra_indicators.get('halftrend_4h'))
            else:
                df['HALFTREND_4H'] = np.nan

        # Feature list extended with new indicators
        features = ['RSI','RSI_fast','RSI_slow','MACD','MACD_signal','MACD_hist',
                    'BB_width','BB_position','ATR','Volatility_20',
                    'Momentum_5','Momentum_10','Momentum_20','Volume_Ratio',
                    'Z_Score_Kalman','Kalman_Slope_sm','Higher_High','Lower_Low',
                    'Price_vs_SMA20','Price_vs_SMA50','Log_Return']

        # Add extra feature names if present
        if extra_indicators:
            if extra_indicators.get('rsi_1h') is not None: features.append('RSI_1H')
            if extra_indicators.get('atr_1h') is not None: features.append('ATR_1H')
            if extra_indicators.get('mfi_1h') is not None: features.append('MFI_1H')
            if extra_indicators.get('halftrend_4h') is not None: features.append('HALFTREND_4H')

        df_ml = df.dropna().copy()
        if len(df_ml) < 60: return 50.0,"NIEDRIG","NEUTRAL",50.0,None

        X=df_ml[features].values; y=df_ml['Target'].values
        fold,acc_lst,sc_wf = len(X)//4,[],RobustScaler()
        for i in range(3):
            te=fold*(i+2); ts=te; te2=min(ts+fold,len(X))
            if te2<=ts: break
            rf_wf=RandomForestClassifier(n_estimators=100,max_depth=6,min_samples_leaf=10,random_state=42)
            rf_wf.fit(sc_wf.fit_transform(X[:te]),y[:te])
            acc_lst.append(accuracy_score(y[ts:te2],rf_wf.predict(sc_wf.transform(X[ts:te2]))))
        treff = np.mean(acc_lst)*100 if acc_lst else 50.0

        sc=RobustScaler(); Xs=sc.fit_transform(X)
        rf=RandomForestClassifier(n_estimators=200,max_depth=6,min_samples_leaf=10,max_features='sqrt',random_state=42)
        gb=GradientBoostingClassifier(n_estimators=150,max_depth=4,learning_rate=0.05,min_samples_leaf=10,subsample=0.8,random_state=42)
        mdl=VotingClassifier(estimators=[('rf',rf),('gb',gb)],voting='soft')
        mdl.fit(Xs,y)

        latest = df[features].dropna().tail(1).values
        latest_s = sc.transform(latest)
        prob_up=float(mdl.predict_proba(latest_s)[0,1]*100)
        abstand=abs(prob_up-50)
        konf="HOCH" if abstand>=20 else ("MITTEL" if abstand>=10 else "NIEDRIG")
        konf_ok=prob_up>=kf_filter or prob_up<=(100-kf_filter)
        if   prob_up>=threshold and konf_ok:       sig="BUY"
        elif prob_up<=(100-threshold) and konf_ok: sig="SELL"
        else:                                       sig="NEUTRAL"
        atr_val=float(df['ATR'].dropna().values[-1]) if 'ATR' in df.columns else None
        return prob_up,konf,sig,treff,atr_val
    except Exception:
        logger.exception("berechne_signal Fehler")
        return 50.0,"FEHLER","NEUTRAL",50.0,None

# -------------------------
# Backtest engine (mit Max Drawdown und MAE)
# -------------------------
def backtest(df, signal_threshold, kf_filter, sl_mult, tp_mult, startkapital=10000):
    try:
        if len(df) < 60: return None
        df = df.copy()

        # Kalman + Basisindikatoren
        kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1],
                          initial_state_mean=float(df['Close'].values[0]),
                          initial_state_covariance=1, observation_covariance=2, transition_covariance=0.05)
        sm, _ = kf.filter(df['Close'].values)
        df['Kalman_Price'] = np.asarray(sm).ravel()
        df['Kalman_Slope'] = pd.Series(df['Kalman_Price']).diff().values
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

        # Einfaches Modell (wie vorher)
        features = ['RSI', 'ATR']
        df_ml = df.dropna().copy()
        if len(df_ml) < 60: return None
        X = df_ml[features].values; y = df_ml['Target'].values
        sc = RobustScaler(); Xs = sc.fit_transform(X)
        rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=10, max_features='sqrt', random_state=42)
        gb = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, min_samples_leaf=10, subsample=0.8, random_state=42)
        mdl = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
        mdl.fit(Xs, y)
        probs = mdl.predict_proba(Xs)[:, 1] * 100
        df_ml = df_ml.copy()
        df_ml['Prob_Up'] = probs
        df_ml['KI_Signal'] = np.where(
            (df_ml['Prob_Up'] >= signal_threshold) | ((100 - df_ml['Prob_Up']) >= signal_threshold),
            np.where(df_ml['Prob_Up'] >= signal_threshold, 1, -1), 0
        )

        kapital = float(startkapital)
        position = 0.0
        entry_price = None
        entry_idx = None
        sl_preis = 0.0
        tp_preis = 0.0
        trades = []
        equity = [kapital]

        trade_mae_pct_list = []   # MAE in Prozent pro geschlossenem Trade
        trade_mae_abs_list = []   # MAE in $ pro geschlossenem Trade

        for idx in range(len(df_ml) - 1):
            row = df_ml.iloc[idx]
            preis = float(row['Close'])
            atr_val = float(row['ATR']) if not np.isnan(row['ATR']) else preis * 0.02
            naechster = float(df_ml.iloc[idx + 1]['Close'])

            # Wenn Position offen: prüfen SL/TP anhand der nächsten Kerze
            if position != 0:
                hit_sl = (position > 0 and naechster <= sl_preis) or (position < 0 and naechster >= sl_preis)
                hit_tp = (position > 0 and naechster >= tp_preis) or (position < 0 and naechster <= tp_preis)
                if hit_sl or hit_tp:
                    exit_p = sl_preis if hit_sl else tp_preis
                    pnl = (exit_p - entry_price) * position if position > 0 else (entry_price - exit_p) * abs(position)
                    kapital += pnl
                    grund = "SL" if hit_sl else "TP"
                    trades.append({"Datum": str(df_ml.iloc[idx + 1]['Date']), "Aktion": "SELL/CLOSE", "Preis": f"{exit_p:.6f}", "PnL": f"{pnl:+.2f}$", "Grund": grund})

                    # MAE für diesen Trade berechnen (adverse excursion während Trade)
                    try:
                        slice_prices = df_ml['Close'].iloc[entry_idx:idx + 2].astype(float).values
                        if position > 0:
                            mae_abs = float(np.min(slice_prices - entry_price))
                        else:
                            mae_abs = float(np.max(slice_prices - entry_price))
                        mae_pct = (mae_abs / entry_price) * 100
                        trade_mae_abs_list.append(abs(mae_abs))
                        trade_mae_pct_list.append(abs(mae_pct))
                    except Exception:
                        pass

                    position = 0.0; entry_price = None; entry_idx = None; sl_preis = 0.0; tp_preis = 0.0
                    equity.append(kapital)
                    continue

            sig_val = int(row['KI_Signal'])
            if sig_val == 1 and position <= 0:
                if position < 0:
                    pnl = (entry_price - preis) * abs(position); kapital += pnl
                    trades.append({"Datum": str(row['Date']), "Aktion": "CLOSE SHORT", "Preis": f"{preis:.6f}", "PnL": f"{pnl:+.2f}$", "Grund": "Signal"})
                menge = (kapital * 0.95) / preis
                position = menge; entry_price = preis; entry_idx = idx
                sl_preis = preis - sl_mult * atr_val
                tp_preis = preis + tp_mult * atr_val
                trades.append({"Datum": str(row['Date']), "Aktion": "BUY", "Preis": f"{preis:.6f}", "PnL": "-", "Grund": "KI Signal"})
            elif sig_val == -1 and position >= 0:
                if position > 0:
                    pnl = (preis - entry_price) * position; kapital += pnl
                    trades.append({"Datum": str(row['Date']), "Aktion": "CLOSE LONG", "Preis": f"{preis:.6f}", "PnL": f"{pnl:+.2f}$", "Grund": "Signal"})
                    # MAE für geschlossenen Long
                    try:
                        slice_prices = df_ml['Close'].iloc[entry_idx:idx + 1].astype(float).values
                        mae_abs = float(np.min(slice_prices - entry_price))
                        mae_pct = (mae_abs / entry_price) * 100
                        trade_mae_abs_list.append(abs(mae_abs))
                        trade_mae_pct_list.append(abs(mae_pct))
                    except Exception:
                        pass
                    position = 0.0; entry_price = None; entry_idx = None

            equity.append(kapital + (preis - entry_price) * position if position != 0 else kapital)

        # Endwerte
        final_val = equity[-1]
        bnh_val = startkapital * (float(df_ml['Close'].values[-1]) / float(df_ml['Close'].values[0]))
        rendite_pct = (final_val - startkapital) / startkapital * 100
        bnh_pct = (bnh_val - startkapital) / startkapital * 100
        sell_trades = [t for t in trades if "CLOSE" in t["Aktion"] or t["Aktion"] == "SELL/CLOSE"]
        pnls = [float(t["PnL"].replace("$", "").replace("+", "")) for t in sell_trades if t["PnL"] != "-"]
        gewinner = sum(1 for p in pnls if p > 0)
        verlierer = len(pnls) - gewinner

        # Max Drawdown (Equity) - Peak to trough in Prozent (negativ)
        eq_arr = np.array(equity)
        running_max = np.maximum.accumulate(eq_arr)
        drawdowns = (eq_arr - running_max) / running_max * 100
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Maximaler Trade Drawdown (MAE) - größter adverse excursion
        max_trade_mae_pct = float(max(trade_mae_pct_list)) if trade_mae_pct_list else 0.0
        max_trade_mae_abs = float(max(trade_mae_abs_list)) if trade_mae_abs_list else 0.0

        # Schutz gegen NaN/Inf
        if not np.isfinite(max_dd): max_dd = 0.0
        if not np.isfinite(max_trade_mae_pct): max_trade_mae_pct = 0.0
        if not np.isfinite(max_trade_mae_abs): max_trade_mae_abs = 0.0

        return {
            "final_val": final_val,
            "bnh_val": bnh_val,
            "rendite_pct": rendite_pct,
            "bnh_pct": bnh_pct,
            "trades": trades,
            "equity": equity,
            "gewinner": gewinner,
            "verlierer": verlierer,
            "max_dd": max_dd,                         # z.B. -12.34 (%)
            "max_trade_mae_pct": max_trade_mae_pct,   # z.B. 5.12 (%)
            "max_trade_mae_abs": max_trade_mae_abs,   # z.B. 123.45 ($)
            "n_trades": len(sell_trades),
            "dates": list(df_ml['Date'].astype(str)),
        }
    except Exception:
        logger.exception("backtest failed")
        return None

# -------------------------
# Session state init
# -------------------------
defaults = {
    "extra_assets":      {},
    "letzte_signale_1h": {},
    "letzte_signale_1d": {},
    "signal_log":        [],
    "farb_index":        0,
    "letzter_refresh_1h":0,
    "letzter_refresh_1d":0,
    "cache_signale_1h":  {},
    "cache_signale_1d":  {},
    "cache_preise":      {},
    "cache_dfs_1h":      {},
    "cache_dfs_1d":      {},
    "neue_assets_queue": [],
}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k]=v

# -------------------------
# Sidebar: categories + search + settings
# -------------------------
st.sidebar.header("Kategorien & Schnellwahl")
with st.sidebar.expander("Crypto (Beispiele)"):
    for s in EXAMPLES_CRYPTO:
        already = s in st.session_state.extra_assets or s in COINS_FEST
        if not already:
            if st.button(f"+ {s}", key=f"add_crypto_{s}", use_container_width=True):
                farbe = FARBEN_EXTRA[st.session_state.farb_index % len(FARBEN_EXTRA)]
                st.session_state.extra_assets[s] = {"yf":s,"name":s,"farbe":farbe,"typ":"Krypto"}
                st.session_state.farb_index += 1
                if s not in st.session_state.neue_assets_queue: st.session_state.neue_assets_queue.append(s)
                st.experimental_rerun()
        else:
            st.caption(f"[OK] {s}")

with st.sidebar.expander("Forex (Beispiele)"):
    for s in EXAMPLES_FOREX:
        already = s in st.session_state.extra_assets or s in COINS_FEST
        if not already:
            if st.button(f"+ {s}", key=f"add_fx_{s}", use_container_width=True):
                farbe = FARBEN_EXTRA[st.session_state.farb_index % len(FARBEN_EXTRA)]
                st.session_state.extra_assets[s] = {"yf":s,"name":s,"farbe":farbe,"typ":"Forex"}
                st.session_state.farb_index += 1
                if s not in st.session_state.neue_assets_queue: st.session_state.neue_assets_queue.append(s)
                st.experimental_rerun()
        else:
            st.caption(f"[OK] {s}")

with st.sidebar.expander("Aktien (Beispiele)"):
    for s in EXAMPLES_STOCKS:
        already = s in st.session_state.extra_assets or s in COINS_FEST
        if not already:
            if st.button(f"+ {s}", key=f"add_stock_{s}", use_container_width=True):
                farbe = FARBEN_EXTRA[st.session_state.farb_index % len(FARBEN_EXTRA)]
                st.session_state.extra_assets[s] = {"yf":s,"name":s,"farbe":farbe,"typ":"Aktie"}
                st.session_state.farb_index += 1
                if s not in st.session_state.neue_assets_queue: st.session_state.neue_assets_queue.append(s)
                st.experimental_rerun()
        else:
            st.caption(f"[OK] {s}")

st.sidebar.markdown("---")
st.sidebar.subheader("Symbol suchen und hinzufügen")
such_symbol = st.sidebar.text_input("Symbol eingeben (z.B. AAPL oder BTC-USD)", placeholder="z.B. AAPL oder BTC-USD")
such_typ    = st.sidebar.selectbox("Typ", ["Krypto","Forex","Aktie"], index=0)
if st.sidebar.button("Symbol hinzufügen"):
    if such_symbol:
        s = yf_sym(such_symbol, such_typ)
        with st.sidebar.status(f"Prüfe {s}..."):
            if validiere(s):
                extras = compute_extra_indicators(s)
                if extras['rsi_1h'] is None or extras['atr_1h'] is None:
                    st.sidebar.error(f"{s}: 1h Indikatoren nicht verfügbar. Hinzufügen abgebrochen.")
                else:
                    farbe = FARBEN_EXTRA[st.session_state.farb_index % len(FARBEN_EXTRA)]
                    st.session_state.extra_assets[s] = {"yf":s,"name":s,"farbe":farbe,"typ":such_typ}
                    st.session_state.farb_index += 1
                    if s not in st.session_state.neue_assets_queue: st.session_state.neue_assets_queue.append(s)
                    st.sidebar.success(f"{s} hinzugefügt!")
                    st.experimental_rerun()
            else:
                st.sidebar.error(f"'{s}' nicht gefunden oder nicht genügend Daten.")

st.sidebar.markdown("---")
st.sidebar.header("Einstellungen")
tg_token   = st.sidebar.text_input("Bot-Token:", type="password")
tg_chat_id = st.sidebar.text_input("Chat-ID:")
auto_refresh = st.sidebar.checkbox("Auto-Refresh aktiv", value=True)
refresh_1h_min = st.sidebar.slider("1h Intervall (Min)", 5, 60, 15)
refresh_1d_min = st.sidebar.slider("1d Intervall (Min)", 30, 360, 60)
REFRESH_1H = refresh_1h_min * 60
REFRESH_1D = refresh_1d_min * 60
threshold = st.sidebar.slider("KI Schwellenwert (%)", 50, 75, 55)
kf_filter = st.sidebar.slider("Mindest-Konfidenz (%)", 50, 80, 60)
sl_mult = st.sidebar.slider("Stop-Loss (ATR x)", 0.5, 3.0, 1.5, 0.1)
tp_mult = st.sidebar.slider("Take-Profit (ATR x)", 1.0, 5.0, 2.5, 0.1)

# -------------------------
# Compute signals for watchlist (with extra indicators)
# -------------------------
alle_assets = {**COINS_FEST, **st.session_state.extra_assets}
jetzt   = time.time()
soll_1h = (jetzt - st.session_state.letzter_refresh_1h) >= REFRESH_1H
soll_1d = (jetzt - st.session_state.letzter_refresh_1d) >= REFRESH_1D

neue_noch_nicht_berechnet = [s for s in st.session_state.neue_assets_queue if s not in st.session_state.cache_signale_1h]
assets_zu_berechnen = []
if soll_1h or soll_1d:
    assets_zu_berechnen = list(alle_assets.keys())
elif neue_noch_nicht_berechnet:
    assets_zu_berechnen = neue_noch_nicht_berechnet

if assets_zu_berechnen:
    fortschritt = st.progress(0, text="Berechne Signale...")
    n = len(assets_zu_berechnen)
    for i, sym in enumerate(assets_zu_berechnen):
        info = alle_assets.get(sym, {"name": sym})
        fortschritt.progress((i+1)/n, text=f"Berechne {info['name']} ({i+1}/{n})...")
        extras = compute_extra_indicators(sym)
        for interval, days, cache_sig, cache_df in [("1h",90,"cache_signale_1h","cache_dfs_1h"),("1d",365,"cache_signale_1d","cache_dfs_1d")]:
            if not soll_1h and not soll_1d and interval=="1d" and sym not in neue_noch_nicht_berechnet: continue
            if not soll_1d and interval=="1d" and sym not in neue_noch_nicht_berechnet: continue
            df = lade_daten(sym, interval, days)
            if df.empty or len(df) < 60: continue
            st.session_state[cache_df][sym] = df
            if sym not in st.session_state.cache_preise or interval=="1h":
                st.session_state.cache_preise[sym] = float(df['Close'].values[-1])
            prob, konf, sig, treff, atr = berechne_signal(df.copy(), threshold, kf_filter, extra_indicators=extras)
            sl, tp, sl_pct, tp_pct = berechne_sl_tp(st.session_state.cache_preise[sym], sig, atr, sl_mult, tp_mult)
            st.session_state[cache_sig][sym] = {"prob":prob,"konf":konf,"signal":sig,"trefferquote":treff,"sl":sl,"tp":tp,"sl_pct":sl_pct,"tp_pct":tp_pct}
    fortschritt.empty()
    if soll_1h: st.session_state.letzter_refresh_1h = jetzt
    if soll_1d: st.session_state.letzter_refresh_1d = jetzt
    st.session_state.neue_assets_queue = [s for s in st.session_state.neue_assets_queue if s not in st.session_state.cache_signale_1h]

signale_1h = st.session_state.cache_signale_1h
signale_1d = st.session_state.cache_signale_1d
preise     = st.session_state.cache_preise
dfs_1h     = st.session_state.cache_dfs_1h
dfs_1d     = st.session_state.cache_dfs_1d

# -------------------------
# Chart function: candlestick + Kalman + SL/TP + trade markers
# -------------------------
def zeige_chart_candles(df_c, info, trades=None):
    if df_c is None or df_c.empty:
        st.warning("Keine Chart-Daten"); return
    df_p = df_c.tail(500).copy()
    try:
        kf = KalmanFilter(transition_matrices=[1],observation_matrices=[1],
            initial_state_mean=float(df_p['Close'].values[0]),initial_state_covariance=1,observation_covariance=2,transition_covariance=0.05)
        sm,_ = kf.filter(df_p['Close'].values)
        df_p['Kalman'] = np.asarray(sm).ravel()
    except Exception:
        df_p['Kalman'] = df_p['Close'].rolling(3, min_periods=1).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df_p['Date'], open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'],
                                 increasing_line_color=info.get('farbe','#00ff88'), decreasing_line_color='#888888', name='Kurs'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_p['Date'], y=df_p['Kalman'], mode='lines', line=dict(color='#00BFFF', width=2), name='Kalman'), row=1, col=1)

    sl = info.get('sl'); tp = info.get('tp')
    if sl:
        fig.add_hline(y=sl, line_dash='dash', line_color='#cc3333', annotation_text=f"SL {sl:.6f}", row=1, col=1)
    if tp:
        fig.add_hline(y=tp, line_dash='dash', line_color='#00aa44', annotation_text=f"TP {tp:.6f}", row=1, col=1)

    if trades:
        buy_x, buy_y, sell_x, sell_y = [], [], [], []
        for t in trades:
            try:
                d_raw = t.get('Datum')
                try:
                    dt = pd.to_datetime(d_raw)
                except Exception:
                    dt = d_raw
                p_raw = t.get('Preis')
                try:
                    p = float(str(p_raw).replace('$',''))
                except Exception:
                    p = None
                act = t.get('Aktion','').upper()
                if p is None: continue
                if 'BUY' in act and 'CLOSE' not in act:
                    buy_x.append(dt); buy_y.append(p)
                elif 'SELL' in act or 'CLOSE' in act:
                    sell_x.append(dt); sell_y.append(p)
            except Exception:
                continue
        if buy_x:
            fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers', marker_symbol='triangle-up', marker_color='#00ff88', marker_size=10, name='Buy'), row=1, col=1)
        if sell_x:
            fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers', marker_symbol='triangle-down', marker_color='#ff6b6b', marker_size=10, name='Sell'), row=1, col=1)

    fig.add_trace(go.Bar(x=df_p['Date'], y=df_p['Volume'], marker_color='rgba(150,150,150,0.3)', name='Vol'), row=2, col=1)
    fig.update_layout(height=650, hovermode='x unified', xaxis_rangeslider_visible=False, showlegend=True, margin=dict(t=20,b=10))
    fig.update_yaxes(title_text='Preis', row=1, col=1)
    fig.update_yaxes(title_text='Vol', row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# UI: main layout (watchlist left, detail right)
# -------------------------
st.title("[CHART] Multi-Asset KI-Signal Dashboard + Backtest")
left_col, right_col = st.columns([1,3])

with left_col:
    st.subheader("Watchlist")
    for sym, info in alle_assets.items():
        with st.container():
            st.markdown(f"<div class='card'><b>{info.get('name',sym)}</b> <span class='small-muted'>({info.get('typ','-')})</span></div>", unsafe_allow_html=True)
            c1, c2 = st.columns([3,1])
            if c2.button("X", key=f"rm_{sym}"):
                if sym in st.session_state.extra_assets: del st.session_state.extra_assets[sym]
                for cache in ["cache_signale_1h","cache_signale_1d","cache_preise","cache_dfs_1h","cache_dfs_1d"]:
                    st.session_state[cache].pop(sym, None)
                st.experimental_rerun()
    st.markdown("---")
    st.caption("Suche oben in der Sidebar oder nutze die Schnellwahl.")

with right_col:
    st.subheader("Detail Analyse / Signale / Backtest")
    all_symbols = list(alle_assets.keys())
    if not all_symbols:
        st.info("Keine Symbole in der Watchlist. Füge links welche hinzu.")
    else:
        selected = st.selectbox("Symbol wählen für Analyse", all_symbols, index=0)
        info = alle_assets.get(selected, {"name":selected,"farbe":"#3498DB","typ":"-"})
        price = preise.get(selected, None)
        sig1 = signale_1h.get(selected, {}).get("signal", "-")
        sig2 = signale_1d.get(selected, {}).get("signal", "-")
        st.markdown(f"### {info['name']}  <span class='small-muted'>({info['typ']})</span>", unsafe_allow_html=True)
        st.write(f"Preis: **{price}**  |  1h: **{sig1}**  |  1d: **{sig2}**")

        # show signal cards
        c1, c2 = st.columns(2)
        def show_card(col, label, sig_info):
            with col:
                st.markdown(f"#### {label}")
                if not sig_info:
                    st.info("Wird berechnet...")
                else:
                    sig = sig_info.get('signal','-')
                    cls = "signal-neutral"
                    if sig == "BUY": cls = "signal-buy"
                    if sig == "SELL": cls = "signal-sell"
                    st.markdown(f"<div class='card'><div class='{cls}' style='font-size:18px'>{signal_label(sig)}</div>"
                                f"<div class='small-muted'>Prob: {sig_info.get('prob',0):.1f}% | Konf: {sig_info.get('konf','-')}</div>"
                                f"<div class='small-muted'>KI-Genauigkeit: {sig_info.get('trefferquote',0):.1f}%</div>"
                                f"<div style='margin-top:8px'>SL: {sig_info.get('sl')} | TP: {sig_info.get('tp')}</div></div>", unsafe_allow_html=True)
                    if st.button(f"Senden {label}", key=f"send_{selected}_{label}"):
                        ok,msg = sende_signal_beide(tg_token, tg_chat_id, info['name'], selected, sig_info.get('signal'), price, sig_info.get('prob'), sig_info.get('konf'), label, sig_info.get('trefferquote'), sig_info.get('sl'), sig_info.get('tp'), sig_info.get('sl_pct'), sig_info.get('tp_pct'), info.get('typ','-'))
                        st.success("Gesendet!") if ok else st.error(msg)
        show_card(c1, "1h", signale_1h.get(selected, {}))
        show_card(c2, "1d", signale_1d.get(selected, {}))

        st.markdown("---")
        st.subheader("Charts (Candles + Kalman + SL/TP)")
        zeige_chart_candles(dfs_1h.get(selected), {**info, **(signale_1h.get(selected, {}))}, trades=None)
        st.markdown("---")

        # Backtest UI
        st.subheader("Backtest deiner Signale")
        bt_interval = st.radio("Zeitrahmen für Backtest", ["1h","1d"], index=0, horizontal=True)
        bt_kapital = st.number_input("Startkapital ($)", min_value=100, value=10000, step=1000)
        bt_sl_mult = st.number_input("SL (ATR x)", min_value=0.1, value=sl_mult, step=0.1)
        bt_tp_mult = st.number_input("TP (ATR x)", min_value=0.1, value=tp_mult, step=0.1)
        if st.button("Backtest starten", key=f"bt_run_{selected}"):
            bt_df = dfs_1h.get(selected) if bt_interval=="1h" else dfs_1d.get(selected)
            if bt_df is None or bt_df.empty:
                st.warning("Keine Daten für Backtest vorhanden.")
            else:
                with st.spinner("Berechne Backtest..."):
                    result = backtest(bt_df, threshold, kf_filter, bt_sl_mult, bt_tp_mult, bt_kapital)
                if result is None:
                    st.error("Backtest fehlgeschlagen.")
                else:
                    m1,m2,m3,m4,m5 = st.columns(5)
                    m1.metric("KI-Strategie", f"${result['final_val']:,.2f}", delta=f"{result['rendite_pct']:+.2f}%")
                    m2.metric("Buy & Hold", f"${result['bnh_val']:,.2f}", delta=f"{result['bnh_pct']:+.2f}%")
                    m3.metric("Trades", result['n_trades'])
                    m4.metric("Gewinner / Verlierer", f"{result['gewinner']} / {result['verlierer']}")
                    m5.metric("Max. Drawdown (Equity)", f"{result['max_dd']:.2f}%")
                    st.metric("Max. Trade Drawdown (MAE %)", f"{result['max_trade_mae_pct']:.2f}%")
                    st.metric("Max. Trade Drawdown ($)", f"${result['max_trade_mae_abs']:.2f}")
                    eq_fig = go.Figure()
                    eq_fig.add_trace(go.Scatter(y=result['equity'], name='KI-Strategie', line=dict(color=info.get('farbe','#00ff88'), width=2)))
                    bnh_equity = [bt_kapital*(float(bt_df['Close'].values[min(i,len(bt_df)-1)])/float(bt_df['Close'].values[0])) for i in range(len(result['equity']))]
                    eq_fig.add_trace(go.Scatter(y=bnh_equity, name='Buy & Hold', line=dict(color='#888888', dash='dash', width=1.5)))
                    eq_fig.update_layout(title=f"Equity-Kurve {selected} ({bt_interval})", height=350, hovermode='x unified', margin=dict(t=40,b=20))
                    st.plotly_chart(eq_fig, use_container_width=True)
                    zeige_chart_candles(bt_df, {**info, **(signale_1h.get(selected, {}))}, trades=result.get('trades', []))
                    if result['trades']:
                        st.markdown("**Trade-Protokoll:**")
                        tr_df = pd.DataFrame(result['trades'])
                        st.dataframe(tr_df, use_container_width=True, height=220)

# -------------------------
# Signal log and summary
# -------------------------
st.markdown("---")
st.subheader("Signal-Wechsel Protokoll")
if st.session_state.signal_log:
    log_df = pd.DataFrame(st.session_state.signal_log[::-1])
    st.dataframe(log_df, use_container_width=True, height=220)
    if st.button("Log leeren"):
        st.session_state.signal_log = []; st.experimental_rerun()
else:
    st.info("Noch keine Signal-Wechsel.")

if st.button("Zusammenfassung senden (alle Assets)"):
    if tg_token and tg_chat_id:
        z = [f"*Multi-Asset Update*\n{'-'*34}"]
        for sym, ai in alle_assets.items():
            p = preise.get(sym, 0)
            sh = st.session_state.cache_signale_1h.get(sym, {}).get("signal", "-")
            sd = st.session_state.cache_signale_1d.get(sym, {}).get("signal", "-")
            z.append(f"{ai['name']:12} ${p:>10,.2f}\n  1h: {sh:7} | 1d: {sd}")
        z.append(f"{'-'*34}\n{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
        ok,msg = sende_telegram(tg_token, tg_chat_id, "\n".join(z))
        st.success("Gesendet!") if ok else st.error(msg)
    else:
        st.error("Telegram Token und Chat-ID erforderlich.")

# -------------------------
# Auto-refresh
# -------------------------
if auto_refresh:
    naechster = min(
        max(0,int(REFRESH_1H-(time.time()-st.session_state.letzter_refresh_1h))),
        max(0,int(REFRESH_1D-(time.time()-st.session_state.letzter_refresh_1d)))
    )
    if naechster <= 0:
        st.experimental_rerun()
    else:
        time.sleep(min(naechster, 30))
        st.experimental_rerun()
