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
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURATION
# ============================================================
st.set_page_config(page_title="Signal Dashboard", page_icon="[CHART]", layout="wide")
st.title("[CHART] Multi-Asset KI-Signal Dashboard")
st.markdown("BUY/SELL-Signale fuer Krypto, Aktien und ETFs | 1h + 1d | Telegram | Backtest")

COINS_FEST = {
    "BTC-USD": {"yf":"BTC-USD","name":"Bitcoin", "farbe":"#F7931A","typ":"Krypto"},
    "ETH-USD": {"yf":"ETH-USD","name":"Ethereum","farbe":"#627EEA","typ":"Krypto"},
    "BNB-USD": {"yf":"BNB-USD","name":"BNB",     "farbe":"#F3BA2F","typ":"Krypto"},
}

BEISPIELE = {
    "Krypto": ["SOL-USD","XRP-USD","DOGE-USD","ADA-USD","AVAX-USD"],
    "Aktien": ["AAPL","TSLA","MSFT","NVDA","AMZN"],
    "ETFs":   ["SPY","QQQ","VTI","GLD","ARKK"],
}

FARBEN_EXTRA = [
    "#E74C3C","#8E44AD","#2ECC71","#1ABC9C","#3498DB",
    "#F39C12","#D35400","#C0392B","#16A085","#27AE60",
]

# ============================================================
# HILFSFUNKTIONEN
# ============================================================
def richtungs_prob(prob):
    return f"{prob:.1f}% Aufwaerts" if prob >= 50 else f"{100-prob:.1f}% Abwaerts"

def signal_farbe(sig):
    return {"BUY":"#00aa44","SELL":"#cc0000"}.get(sig,"#888888")

def signal_label(sig):
    return {"BUY":"BUY / KAUFEN","SELL":"SELL / VERKAUFEN"}.get(sig,"NEUTRAL / ABWARTEN")

def berechne_sl_tp(preis, sig, atr, sl_mult, tp_mult):
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

# ============================================================
# TELEGRAM
# ============================================================
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
        return False, str(e)

def tg_nachricht(name, sym, sig, preis, prob, konf, zr, treff, sl, tp, sl_pct, tp_pct, typ):
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

# ============================================================
# DATEN LADEN
# ============================================================
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
    except: return pd.DataFrame()

def validiere(symbol):
    df = lade_daten(symbol,"1d",10)
    return not df.empty and len(df) >= 3

def berechne_ein_asset(yf_sym_str, threshold, kf_filter, sl_mult, tp_mult):
    """Laedt und berechnet 1h+1d Signal fuer ein einzelnes Asset."""
    ergebnis = {}
    for interval, days, key in [("1h",90,"1h"),("1d",365,"1d")]:
        df = lade_daten(yf_sym_str, interval, days)
        if df.empty or len(df) < 60:
            ergebnis[key] = {"df": None, "signal":{}}
            continue
        preis = float(df['Close'].values[-1])
        prob, konf, sig, treff, atr = berechne_signal(df.copy(), threshold, kf_filter)
        sl, tp, sl_pct, tp_pct = berechne_sl_tp(preis, sig, atr, sl_mult, tp_mult)
        ergebnis[key] = {
            "df": df, "preis": preis,
            "signal": {"prob":prob,"konf":konf,"signal":sig,
                       "trefferquote":treff,"sl":sl,"tp":tp,
                       "sl_pct":sl_pct,"tp_pct":tp_pct}
        }
    return ergebnis

# ============================================================
# KI SIGNAL ENGINE
# ============================================================
def berechne_signal(df, threshold, kf_filter):
    try:
        if len(df) < 60: return 50.0,"NIEDRIG","NEUTRAL",50.0,None
        kf = KalmanFilter(transition_matrices=[1],observation_matrices=[1],
            initial_state_mean=float(df['Close'].values[0]),
            initial_state_covariance=1,observation_covariance=2,transition_covariance=0.05)
        sm,_ = kf.filter(df['Close'].values)
        df['Kalman_Price'] = sm
        df['Kalman_Slope'] = pd.Series(sm.flatten()).diff().values
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
        features = ['RSI','RSI_fast','RSI_slow','MACD','MACD_signal','MACD_hist',
                    'BB_width','BB_position','ATR','Volatility_20',
                    'Momentum_5','Momentum_10','Momentum_20','Volume_Ratio',
                    'Z_Score_Kalman','Kalman_Slope_sm','Higher_High','Lower_Low',
                    'Price_vs_SMA20','Price_vs_SMA50','Log_Return']
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
        latest=sc.transform(df[features].dropna().tail(1).values)
        prob_up=float(mdl.predict_proba(latest)[0,1]*100)
        abstand=abs(prob_up-50)
        konf="HOCH" if abstand>=20 else ("MITTEL" if abstand>=10 else "NIEDRIG")
        konf_ok=prob_up>=kf_filter or prob_up<=(100-kf_filter)
        if   prob_up>=threshold and konf_ok:       sig="BUY"
        elif prob_up<=(100-threshold) and konf_ok: sig="SELL"
        else:                                       sig="NEUTRAL"
        atr_val=float(df['ATR'].dropna().values[-1]) if 'ATR' in df.columns else None
        return prob_up,konf,sig,treff,atr_val
    except: return 50.0,"FEHLER","NEUTRAL",50.0,None

# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest(df, signal_threshold, kf_filter, sl_mult, tp_mult, startkapital=10000):
    try:
        if len(df) < 60: return None
        # Features berechnen (wiederverwendet berechne_signal intern)
        df = df.copy()
        kf = KalmanFilter(transition_matrices=[1],observation_matrices=[1],
            initial_state_mean=float(df['Close'].values[0]),
            initial_state_covariance=1,observation_covariance=2,transition_covariance=0.05)
        sm,_=kf.filter(df['Close'].values); df['Kalman_Price']=sm
        df['Kalman_Slope']=pd.Series(sm.flatten()).diff().values
        df['RSI']      = ta.rsi(df['Close'],length=14)
        df['RSI_fast'] = ta.rsi(df['Close'],length=7)
        df['RSI_slow'] = ta.rsi(df['Close'],length=21)
        macd=ta.macd(df['Close'],fast=12,slow=26,signal=9)
        if macd is not None and not macd.empty:
            df['MACD']=macd.iloc[:,0]; df['MACD_signal']=macd.iloc[:,1]; df['MACD_hist']=macd.iloc[:,2]
        else: df['MACD']=df['MACD_signal']=df['MACD_hist']=0.0
        bb=ta.bbands(df['Close'],length=20)
        if bb is not None and not bb.empty:
            df['BB_width']=(bb.iloc[:,0]-bb.iloc[:,2])/bb.iloc[:,1]
            df['BB_position']=(df['Close']-bb.iloc[:,2])/(bb.iloc[:,0]-bb.iloc[:,2])
        else: df['BB_width']=df['BB_position']=0.0
        df['ATR']           =ta.atr(df['High'],df['Low'],df['Close'],length=14)
        df['Log_Return']    =np.log(df['Close']/df['Close'].shift(1))
        df['Volatility_20'] =df['Log_Return'].rolling(20).std()
        df['Momentum_5']    =df['Close']/df['Close'].shift(5)-1
        df['Momentum_10']   =df['Close']/df['Close'].shift(10)-1
        df['Momentum_20']   =df['Close']/df['Close'].shift(20)-1
        df['Volume_Ratio']  =df['Volume']/df['Volume'].rolling(20).mean()
        df['Z_Score_Kalman']  =(df['Close']-df['Kalman_Price'])/df['Close'].rolling(20).std()
        df['Kalman_Slope_sm']=pd.Series(df['Kalman_Slope']).rolling(5).mean().values
        df['Price_vs_SMA20']  =df['Close']/df['Close'].rolling(20).mean()-1
        df['Price_vs_SMA50']  =df['Close']/df['Close'].rolling(50).mean()-1
        df['Higher_High']     =(df['High']>df['High'].shift(1)).astype(int)
        df['Lower_Low']       =(df['Low']<df['Low'].shift(1)).astype(int)
        df['Target']          =np.where(df['Close'].shift(-1)>df['Close'],1,0)
        features=['RSI','RSI_fast','RSI_slow','MACD','MACD_signal','MACD_hist',
                  'BB_width','BB_position','ATR','Volatility_20',
                  'Momentum_5','Momentum_10','Momentum_20','Volume_Ratio',
                  'Z_Score_Kalman','Kalman_Slope_sm','Higher_High','Lower_Low',
                  'Price_vs_SMA20','Price_vs_SMA50','Log_Return']
        df_ml=df.dropna().copy()
        if len(df_ml)<60: return None
        X=df_ml[features].values; y=df_ml['Target'].values
        sc=RobustScaler(); Xs=sc.fit_transform(X)
        rf=RandomForestClassifier(n_estimators=200,max_depth=6,min_samples_leaf=10,max_features='sqrt',random_state=42)
        gb=GradientBoostingClassifier(n_estimators=150,max_depth=4,learning_rate=0.05,min_samples_leaf=10,subsample=0.8,random_state=42)
        mdl=VotingClassifier(estimators=[('rf',rf),('gb',gb)],voting='soft')
        mdl.fit(Xs,y)
        probs=mdl.predict_proba(Xs)[:,1]*100
        df_ml=df_ml.copy()
        df_ml['Prob_Up']=probs

        # Signal per Zeitschritt
        df_ml['KI_Signal']=np.where(
            (df_ml['Prob_Up']>=signal_threshold)|((100-df_ml['Prob_Up'])>=signal_threshold),
            np.where(df_ml['Prob_Up']>=signal_threshold,1,-1), 0
        )

        # Trade-Simulation mit SL/TP
        kapital   = float(startkapital)
        position  = 0.0
        ep        = 0.0
        sl_preis  = 0.0
        tp_preis  = 0.0
        trades    = []
        equity    = [kapital]

        for idx in range(len(df_ml)-1):
            row      = df_ml.iloc[idx]
            preis    = float(row['Close'])
            atr_val  = float(row['ATR']) if not np.isnan(row['ATR']) else preis*0.02
            naechster= float(df_ml.iloc[idx+1]['Close'])

            if position != 0:
                # SL/TP pruefen
                hit_sl = (position>0 and naechster<=sl_preis) or (position<0 and naechster>=sl_preis)
                hit_tp = (position>0 and naechster>=tp_preis) or (position<0 and naechster<=tp_preis)
                if hit_sl or hit_tp:
                    exit_p = sl_preis if hit_sl else tp_preis
                    pnl    = (exit_p-ep)*position if position>0 else (ep-exit_p)*abs(position)
                    kapital += pnl
                    grund  = "SL" if hit_sl else "TP"
                    trades.append({"Datum":str(df_ml.iloc[idx+1]['Date'])[:10],
                                   "Aktion":"SELL/CLOSE","Preis":f"${exit_p:,.4f}",
                                   "PnL":f"{pnl:+.2f}$","Grund":grund})
                    position=0.0; ep=0.0; sl_preis=0.0; tp_preis=0.0
                    equity.append(kapital)
                    continue

            sig_val = int(row['KI_Signal'])
            if sig_val == 1 and position <= 0:
                if position < 0:
                    pnl=(ep-preis)*abs(position); kapital+=pnl
                    trades.append({"Datum":str(row['Date'])[:10],"Aktion":"CLOSE SHORT",
                                   "Preis":f"${preis:,.4f}","PnL":f"{pnl:+.2f}$","Grund":"Signal"})
                menge   = (kapital*0.95)/preis
                position= menge; ep=preis
                sl_preis= preis - sl_mult*atr_val
                tp_preis= preis + tp_mult*atr_val
                trades.append({"Datum":str(row['Date'])[:10],"Aktion":"BUY",
                               "Preis":f"${preis:,.4f}","PnL":"-","Grund":"KI Signal"})
            elif sig_val == -1 and position >= 0:
                if position > 0:
                    pnl=(preis-ep)*position; kapital+=pnl
                    trades.append({"Datum":str(row['Date'])[:10],"Aktion":"CLOSE LONG",
                                   "Preis":f"${preis:,.4f}","PnL":f"{pnl:+.2f}$","Grund":"Signal"})
                    position=0.0

            equity.append(kapital + (preis-ep)*position if position!=0 else kapital)

        # Abschluss-Metriken
        final_val   = equity[-1]
        bnh_val     = startkapital*(float(df_ml['Close'].values[-1])/float(df_ml['Close'].values[0]))
        rendite_pct = (final_val-startkapital)/startkapital*100
        bnh_pct     = (bnh_val-startkapital)/startkapital*100
        sell_trades = [t for t in trades if "CLOSE" in t["Aktion"] or t["Aktion"]=="SELL/CLOSE"]
        pnls        = [float(t["PnL"].replace("$","").replace("+","")) for t in sell_trades if t["PnL"]!="-"]
        gewinner    = sum(1 for p in pnls if p>0)
        verlierer   = len(pnls)-gewinner
        max_eq      = max(equity)
        min_nach_max= min(equity[equity.index(max_eq):]) if equity.index(max_eq)<len(equity)-1 else equity[-1]
        max_dd      = (min_nach_max-max_eq)/max_eq*100

        return {
            "final_val":   final_val,
            "bnh_val":     bnh_val,
            "rendite_pct": rendite_pct,
            "bnh_pct":     bnh_pct,
            "trades":      trades,
            "equity":      equity,
            "gewinner":    gewinner,
            "verlierer":   verlierer,
            "max_dd":      max_dd,
            "n_trades":    len(sell_trades),
            "dates":       list(df_ml['Date'].astype(str)),
        }
    except Exception as e:
        return None

# ============================================================
# SESSION STATE
# ============================================================
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
    "neue_assets_queue": [],   # Assets die noch berechnet werden muessen
}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k]=v

# ============================================================
# SEITENLEISTE
# ============================================================
st.sidebar.header("Konfiguration")

# --- Asset hinzufuegen ---
st.sidebar.subheader("Asset hinzufuegen")
st.sidebar.markdown(
    "**Krypto:** `SOL-USD` `XRP-USD` `DOGE-USD`\n\n"
    "**Aktien:** `AAPL` `TSLA` `MSFT` `NVDA`\n\n"
    "**ETFs:** `SPY` `QQQ` `GLD` `ARKK`"
)
neues_symbol = st.sidebar.text_input("Symbol:", placeholder="z.B. AAPL oder SOL-USD").strip().upper()
neuer_typ    = st.sidebar.selectbox("Typ:", ["Krypto","Aktie","ETF"])
neuer_name   = st.sidebar.text_input("Anzeigename (optional):", placeholder="z.B. Apple")

if st.sidebar.button("Hinzufuegen", use_container_width=True):
    if neues_symbol:
        s = yf_sym(neues_symbol, neuer_typ)
        with st.sidebar.status(f"Pruefe {s}..."):
            if validiere(s):
                farbe = FARBEN_EXTRA[st.session_state.farb_index % len(FARBEN_EXTRA)]
                name  = neuer_name if neuer_name else s
                st.session_state.extra_assets[s] = {"yf":s,"name":name,"farbe":farbe,"typ":neuer_typ}
                st.session_state.farb_index += 1
                # NEU: sofort in Queue fuer Berechnung
                if s not in st.session_state.neue_assets_queue:
                    st.session_state.neue_assets_queue.append(s)
                st.sidebar.success(f"{name} hinzugefuegt!")
            else:
                st.sidebar.error(f"'{s}' nicht gefunden.")

# Schnellauswahl
st.sidebar.markdown("**Schnellauswahl:**")
for kat, symbole in BEISPIELE.items():
    with st.sidebar.expander(kat):
        for s in symbole:
            already = s in st.session_state.extra_assets or s in COINS_FEST
            label   = f"[OK] {s}" if already else f"+ {s}"
            if not already:
                if st.button(label, key=f"q_{s}", use_container_width=True):
                    typ = "Krypto" if "-USD" in s else ("ETF" if kat=="ETFs" else "Aktie")
                    farbe = FARBEN_EXTRA[st.session_state.farb_index % len(FARBEN_EXTRA)]
                    st.session_state.extra_assets[s] = {"yf":s,"name":s,"farbe":farbe,"typ":typ}
                    st.session_state.farb_index += 1
                    if s not in st.session_state.neue_assets_queue:
                        st.session_state.neue_assets_queue.append(s)
            else:
                st.sidebar.caption(label)

# Aktive Extra-Assets verwalten
if st.session_state.extra_assets:
    st.sidebar.markdown("**Aktive Extra-Assets:**")
    to_remove = []
    for sym, info in list(st.session_state.extra_assets.items()):
        c1, c2 = st.sidebar.columns([3,1])
        c1.caption(f"{info['name']} ({info['typ']})")
        if c2.button("X", key=f"rm_{sym}"):
            to_remove.append(sym)
    for sym in to_remove:
        del st.session_state.extra_assets[sym]
        for cache in ["cache_signale_1h","cache_signale_1d","cache_preise","cache_dfs_1h","cache_dfs_1d"]:
            st.session_state[cache].pop(sym, None)
    if to_remove:
        st.rerun()

st.sidebar.markdown("---")

# --- Parameter ---
st.sidebar.subheader("Signal-Parameter")
threshold        = st.sidebar.slider("KI Schwellenwert (%)",  50, 75, 55)
kf_filter        = st.sidebar.slider("Mindest-Konfidenz (%)", 50, 80, 60)
sl_mult          = st.sidebar.slider("Stop-Loss (ATR x)",     0.5, 3.0, 1.5, 0.1)
tp_mult          = st.sidebar.slider("Take-Profit (ATR x)",   1.0, 5.0, 2.5, 0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Auto-Refresh")
auto_refresh   = st.sidebar.checkbox("Auto-Refresh aktiv", value=True)
refresh_1h_min = st.sidebar.slider("1h-Signale Intervall (Min)", 5, 60, 15)
refresh_1d_min = st.sidebar.slider("1d-Signale Intervall (Min)", 30, 360, 60)
REFRESH_1H     = refresh_1h_min * 60
REFRESH_1D     = refresh_1d_min * 60

st.sidebar.markdown("---")
st.sidebar.subheader("Telegram")
tg_token   = st.sidebar.text_input("Bot-Token:", type="password")
tg_chat_id = st.sidebar.text_input("Chat-ID:")
if tg_token and tg_chat_id:
    if st.sidebar.button("Verbindungstest"):
        ok, msg = sende_telegram(tg_token, tg_chat_id, "Verbindungstest erfolgreich!")
        st.sidebar.success("Verbunden!") if ok else st.sidebar.error(f"Fehler: {msg}")

st.sidebar.subheader("Autopilot")
autopilot    = st.sidebar.checkbox("Senden bei Signal-Wechsel", value=True)
send_neutral = st.sidebar.checkbox("Auch NEUTRAL senden",       value=False)

# ============================================================
# ALLE ASSETS
# ============================================================
alle_assets = {**COINS_FEST, **st.session_state.extra_assets}

# ============================================================
# SIGNALE BERECHNEN
# ============================================================
jetzt   = time.time()
soll_1h = (jetzt - st.session_state.letzter_refresh_1h) >= REFRESH_1H
soll_1d = (jetzt - st.session_state.letzter_refresh_1d) >= REFRESH_1D

# Neue Assets die noch nicht im Cache sind
neue_noch_nicht_berechnet = [
    s for s in st.session_state.neue_assets_queue
    if s not in st.session_state.cache_signale_1h
]

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

        for interval, days, cache_sig, cache_df in [
            ("1h", 90,  "cache_signale_1h", "cache_dfs_1h"),
            ("1d", 365, "cache_signale_1d", "cache_dfs_1d"),
        ]:
            if not soll_1h and not soll_1d and interval == "1d" and sym not in neue_noch_nicht_berechnet:
                continue
            if not soll_1d and interval == "1d" and sym not in neue_noch_nicht_berechnet:
                continue

            df = lade_daten(sym, interval, days)
            if df.empty or len(df) < 60:
                continue
            st.session_state[cache_df][sym] = df
            if sym not in st.session_state.cache_preise or interval == "1h":
                st.session_state.cache_preise[sym] = float(df['Close'].values[-1])
            prob, konf, sig, treff, atr = berechne_signal(df.copy(), threshold, kf_filter)
            sl, tp, sl_pct, tp_pct = berechne_sl_tp(
                st.session_state.cache_preise[sym], sig, atr, sl_mult, tp_mult
            )
            st.session_state[cache_sig][sym] = {
                "prob":prob,"konf":konf,"signal":sig,
                "trefferquote":treff,"sl":sl,"tp":tp,
                "sl_pct":sl_pct,"tp_pct":tp_pct
            }

    fortschritt.empty()
    if soll_1h: st.session_state.letzter_refresh_1h = jetzt
    if soll_1d: st.session_state.letzter_refresh_1d = jetzt
    # Queue leeren
    st.session_state.neue_assets_queue = [
        s for s in st.session_state.neue_assets_queue
        if s not in st.session_state.cache_signale_1h
    ]

signale_1h = st.session_state.cache_signale_1h
signale_1d = st.session_state.cache_signale_1d
preise     = st.session_state.cache_preise
dfs_1h     = st.session_state.cache_dfs_1h
dfs_1d     = st.session_state.cache_dfs_1d

# ============================================================
# AUTOPILOT TELEGRAM
# ============================================================
def autopilot_check(signale, letzte_key, zr_text):
    if not autopilot or not tg_token or not tg_chat_id: return
    letzte = st.session_state[letzte_key]
    for sym, info in signale.items():
        sig  = info["signal"]
        prev = letzte.get(sym, "INITIAL")
        if sig == prev and prev != "INITIAL": continue
        if sig == "NEUTRAL" and not send_neutral:
            letzte[sym] = sig; continue
        ai   = alle_assets.get(sym, {"name":sym,"typ":"-"})
        preis= preise.get(sym, 0)
        ok,_ = sende_telegram(tg_token, tg_chat_id,
            tg_nachricht(ai["name"],sym,sig,preis,info["prob"],info["konf"],
                         zr_text,info["trefferquote"],
                         info["sl"],info["tp"],info["sl_pct"],info["tp_pct"],ai["typ"]))
        if ok:
            st.session_state.signal_log.append({
                "Zeit":datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                "Asset":ai["name"],"Typ":ai["typ"],"Zeitrahmen":zr_text,
                "Von":prev,"Zu":sig,"Kurs":f"${preis:,.4f}",
                "Richtung":richtungs_prob(info["prob"]),"Konfidenz":info["konf"],
                "SL":f"${info['sl']:,.4f} ({info['sl_pct']:+.2f}%)" if info["sl"] else "-",
                "TP":f"${info['tp']:,.4f} ({info['tp_pct']:+.2f}%)" if info["tp"] else "-",
            })
        letzte[sym] = sig

autopilot_check(signale_1h, "letzte_signale_1h", "naechste Stunde (1h)")
autopilot_check(signale_1d, "letzte_signale_1d", "naechsten Tag (1d)")

if st.sidebar.button("Zusammenfassung senden"):
    if tg_token and tg_chat_id:
        z=[f"*Multi-Asset Update*\n{'-'*34}"]
        for sym,ai in alle_assets.items():
            p=preise.get(sym,0); sh=signale_1h.get(sym,{}).get("signal","-"); sd=signale_1d.get(sym,{}).get("signal","-")
            z.append(f"{ai['name']:12} ${p:>10,.2f}\n  1h: {sh:7} | 1d: {sd}")
        z.append(f"{'-'*34}\n{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
        ok,msg=sende_telegram(tg_token,tg_chat_id,"\n".join(z))
        st.sidebar.success("Gesendet!") if ok else st.sidebar.error(msg)

# Countdown
verbl_1h=max(0,int(REFRESH_1H-(time.time()-st.session_state.letzter_refresh_1h)))
verbl_1d=max(0,int(REFRESH_1D-(time.time()-st.session_state.letzter_refresh_1d)))
st.sidebar.markdown("---")
st.sidebar.caption(f"Naechste Aktualisierung:\n1h: {verbl_1h//60}m {verbl_1h%60}s\n1d: {verbl_1d//60}m {verbl_1d%60}s")

# ============================================================
# CHART FUNKTION
# ============================================================
def zeige_chart(df_c, farbe, info):
    if df_c is None or df_c.empty:
        st.warning("Keine Chart-Daten"); return
    df_p = df_c.tail(200).copy()
    kf = KalmanFilter(transition_matrices=[1],observation_matrices=[1],
        initial_state_mean=float(df_p['Close'].values[0]),
        initial_state_covariance=1,observation_covariance=2,transition_covariance=0.05)
    sm,_=kf.filter(df_p['Close'].values); df_p['Kalman']=sm
    df_p['BB_upper']=df_p['Close'].rolling(20).mean()+2*df_p['Close'].rolling(20).std()
    df_p['BB_lower']=df_p['Close'].rolling(20).mean()-2*df_p['Close'].rolling(20).std()
    df_p['RSI']=ta.rsi(df_p['Close'],length=14)
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[0.6,0.2,0.2],vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(x=df_p['Date'],open=df_p['Open'],high=df_p['High'],
        low=df_p['Low'],close=df_p['Close'],name='Kurs',
        increasing_line_color=farbe,decreasing_line_color='#888888'),row=1,col=1)
    fig.add_trace(go.Scatter(x=df_p['Date'],y=df_p['Kalman'],name='Kalman',
        line=dict(color='#00BFFF',width=2)),row=1,col=1)
    fig.add_trace(go.Scatter(x=df_p['Date'],y=df_p['BB_upper'],name='BB Oben',
        line=dict(color='rgba(255,165,0,0.5)',dash='dash',width=1)),row=1,col=1)
    fig.add_trace(go.Scatter(x=df_p['Date'],y=df_p['BB_lower'],name='BB Unten',
        line=dict(color='rgba(255,165,0,0.5)',dash='dash',width=1),
        fill='tonexty',fillcolor='rgba(255,165,0,0.05)'),row=1,col=1)
    sl=info.get("sl"); tp=info.get("tp")
    if sl: fig.add_hline(y=sl,line_dash='dash',line_color='#cc3333',
                         annotation_text=f"SL ${sl:,.2f}",row=1,col=1)
    if tp: fig.add_hline(y=tp,line_dash='dash',line_color='#00aa44',
                         annotation_text=f"TP ${tp:,.2f}",row=1,col=1)
    fig.add_trace(go.Scatter(x=df_p['Date'],y=df_p['RSI'],name='RSI',
        line=dict(color='#9B59B6',width=1.5)),row=2,col=1)
    fig.add_hline(y=70,line_dash='dash',line_color='red',  opacity=0.4,row=2,col=1)
    fig.add_hline(y=30,line_dash='dash',line_color='green',opacity=0.4,row=2,col=1)
    fig.add_trace(go.Bar(x=df_p['Date'],y=df_p['Volume'],name='Vol',
        marker_color='rgba(150,150,150,0.3)'),row=3,col=1)
    fig.update_layout(height=500,hovermode='x unified',xaxis_rangeslider_visible=False,
                      showlegend=False,margin=dict(t=20,b=10))
    fig.update_yaxes(title_text='Preis',row=1,col=1)
    fig.update_yaxes(title_text='RSI',  row=2,col=1)
    fig.update_yaxes(title_text='Vol.', row=3,col=1)
    st.plotly_chart(fig,use_container_width=True)

# ============================================================
# SIGNAL-KARTE FUNKTION
# ============================================================
def zeige_signal_karte(col, info, zr_label, zr_key, farbe, sym, name, typ, preis):
    with col:
        if not info:
            st.info(f"Wird berechnet..."); return
        sig=info.get("signal","NEUTRAL"); prob=info.get("prob",50.0)
        konf=info.get("konf","-"); treff=info.get("trefferquote",50.0)
        sl=info.get("sl"); tp=info.get("tp")
        sl_pct=info.get("sl_pct"); tp_pct=info.get("tp_pct")
        sf=signal_farbe(sig)
        sl_s=f"${sl:,.4f} ({sl_pct:+.2f}%)" if sl else "-"
        tp_s=f"${tp:,.4f} ({tp_pct:+.2f}%)" if tp else "-"
        st.markdown(
            f"<div style='border:2px solid {farbe};border-radius:10px;padding:14px;'>"
            f"<h4 style='margin:0;color:{farbe}'>{zr_label}</h4>"
            f"<p style='font-size:24px;font-weight:bold;color:{sf};margin:6px 0'>{signal_label(sig)}</p>"
            f"<p style='margin:2px 0'>Richtung: <b>{richtungs_prob(prob)}</b></p>"
            f"<p style='margin:2px 0'>Konfidenz: <b>{konf}</b></p>"
            f"<p style='margin:2px 0;font-size:12px;color:gray'>KI-Genauigkeit: {treff:.1f}%</p>"
            f"<hr style='margin:8px 0;border-color:{farbe}44'>"
            f"<p style='margin:2px 0;color:#cc3333'>Stop-Loss:   <b>{sl_s}</b></p>"
            f"<p style='margin:2px 0;color:#00aa44'>Take-Profit: <b>{tp_s}</b></p>"
            f"</div>",unsafe_allow_html=True
        )
        if tg_token and tg_chat_id:
            if st.button(f"Senden ({zr_key})", key=f"btn_{sym}_{zr_key}"):
                ok,msg=sende_telegram(tg_token,tg_chat_id,
                    tg_nachricht(name,sym,sig,preis,prob,konf,zr_label,treff,sl,tp,sl_pct,tp_pct,typ))
                st.success("Gesendet!") if ok else st.error(msg)

# ============================================================
# DASHBOARD
# ============================================================
st.caption(f"Letztes Update: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
st.markdown("---")

for sym, ai in alle_assets.items():
    farbe = ai["farbe"]; name=ai["name"]; typ=ai["typ"]
    preis = preise.get(sym, 0)
    i1h   = signale_1h.get(sym, {})
    i1d   = signale_1d.get(sym, {})

    # Asset-Header
    preis_str = f"${preis:,.4f}" if preis > 0 else "wird geladen..."
    st.markdown(
        f"<h2 style='color:{farbe};border-bottom:2px solid {farbe};padding-bottom:5px'>"
        f"{name} &nbsp;<span style='font-size:13px;color:gray'>({typ})</span>"
        f"&nbsp;&nbsp; {preis_str}"
        f"</h2>",unsafe_allow_html=True
    )

    if preis == 0:
        st.info("Daten werden geladen... Seite wird automatisch aktualisiert.")
        st.markdown("---"); continue

    # Signal-Karten
    c1, c2 = st.columns(2)
    zeige_signal_karte(c1, i1h, "1h - Stunden", "1h", farbe, sym, name, typ, preis)
    zeige_signal_karte(c2, i1d, "1d - Tage",    "1d", farbe, sym, name, typ, preis)

    # Charts
    ct1, ct2 = st.tabs([f"{name} 1h-Chart", f"{name} 1d-Chart"])
    with ct1: zeige_chart(dfs_1h.get(sym), farbe, i1h)
    with ct2: zeige_chart(dfs_1d.get(sym), farbe, i1d)

    # --- BACKTEST (aufklappbar) ---
    with st.expander(f"Backtest {name} anzeigen"):
        bt_col1, bt_col2 = st.columns([2,1])
        with bt_col1:
            bt_interval = st.radio(
                "Zeitrahmen:", ["1h","1d"],
                key=f"bt_iv_{sym}", horizontal=True
            )
        with bt_col2:
            bt_kapital = st.number_input(
                "Startkapital ($)", min_value=100, value=10000, step=1000,
                key=f"bt_kap_{sym}"
            )

        if st.button(f"Backtest starten", key=f"bt_run_{sym}"):
            bt_df = dfs_1h.get(sym) if bt_interval=="1h" else dfs_1d.get(sym)
            if bt_df is None or bt_df.empty:
                st.warning("Keine Daten fuer Backtest vorhanden.")
            else:
                with st.spinner("Berechne Backtest..."):
                    result = backtest(bt_df, threshold, kf_filter, sl_mult, tp_mult, bt_kapital)
                if result is None:
                    st.error("Backtest fehlgeschlagen.")
                else:
                    # Kennzahlen
                    m1,m2,m3,m4,m5 = st.columns(5)
                    m1.metric("KI-Strategie", f"${result['final_val']:,.2f}",
                              delta=f"{result['rendite_pct']:+.2f}%")
                    m2.metric("Buy & Hold",   f"${result['bnh_val']:,.2f}",
                              delta=f"{result['bnh_pct']:+.2f}%")
                    m3.metric("Trades",       result['n_trades'])
                    m4.metric("Gewinner / Verlierer",
                              f"{result['gewinner']} / {result['verlierer']}")
                    m5.metric("Max. Drawdown", f"{result['max_dd']:.2f}%")

                    # Equity-Kurve
                    eq_fig = go.Figure()
                    eq_fig.add_trace(go.Scatter(
                        y=result['equity'],
                        name='KI-Strategie',
                        line=dict(color=farbe, width=2)
                    ))
                    bnh_equity = [bt_kapital*(float(bt_df['Close'].values[min(i,len(bt_df)-1)])/float(bt_df['Close'].values[0]))
                                  for i in range(len(result['equity']))]
                    eq_fig.add_trace(go.Scatter(
                        y=bnh_equity,
                        name='Buy & Hold',
                        line=dict(color='#888888', dash='dash', width=1.5)
                    ))
                    eq_fig.update_layout(
                        title=f"Equity-Kurve {name} ({bt_interval})",
                        height=350, hovermode='x unified',
                        legend=dict(orientation='h',y=1.02,x=1,xanchor='right',yanchor='bottom'),
                        margin=dict(t=40,b=20)
                    )
                    eq_fig.update_yaxes(title_text='Kapital ($)')
                    st.plotly_chart(eq_fig, use_container_width=True)

                    # Trade-Tabelle
                    if result['trades']:
                        st.markdown("**Trade-Protokoll:**")
                        tr_df = pd.DataFrame(result['trades'])
                        def bt_farbe(row):
                            if 'BUY' in row['Aktion']: return ['background-color:rgba(0,180,0,0.1)']*len(row)
                            if 'CLOSE' in row['Aktion'] or 'SELL' in row['Aktion']:
                                return ['background-color:rgba(200,0,0,0.1)']*len(row)
                            return ['']*len(row)
                        st.dataframe(tr_df.style.apply(bt_farbe,axis=1),
                                     use_container_width=True, height=220)

    st.markdown("---")

# ============================================================
# SIGNAL-LOG
# ============================================================
st.subheader("Signal-Wechsel Protokoll")
if st.session_state.signal_log:
    log_df=pd.DataFrame(st.session_state.signal_log[::-1])
    def zf(row):
        c=('rgba(0,180,0,0.12)' if row['Zu']=='BUY'
           else 'rgba(200,0,0,0.12)' if row['Zu']=='SELL' else '')
        return [f'background-color:{c}']*len(row)
    st.dataframe(log_df.style.apply(zf,axis=1),use_container_width=True,height=220)
    if st.button("Log leeren"):
        st.session_state.signal_log=[]; st.rerun()
else:
    st.info("Noch keine Signal-Wechsel.")

# ============================================================
# AUTO-REFRESH
# ============================================================
if auto_refresh:
    naechster=min(
        max(0,int(REFRESH_1H-(time.time()-st.session_state.letzter_refresh_1h))),
        max(0,int(REFRESH_1D-(time.time()-st.session_state.letzter_refresh_1d)))
    )
    if naechster<=0:
        st.rerun()
    else:
        time.sleep(min(naechster,30))
        st.rerun()
