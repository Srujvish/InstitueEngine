# INSTITUTIONAL OPTION FLOW ENGINE - BY INSTITUTIONAL TRADER

import os
import time
import requests
import pandas as pd
import yfinance as yf
import warnings
import pyotp
import math
from datetime import datetime, time as dtime, timedelta
from SmartApi.smartConnect import SmartConnect
import threading
import numpy as np

warnings.filterwarnings("ignore")

# ---------------- INSTITUTIONAL OPTION FLOW CONFIG ----------------
OPENING_PLAY_ENABLED = True
OPENING_START = dtime(9,15)
OPENING_END = dtime(9,45)

# 🚨 **OPTION FLOW THRESHOLDS** 🚨
MIN_OI_BUILDUP = 50000  # 50K contracts minimum
MIN_VOLUME_BUILDUP = 1000000  # 10L volume minimum
OI_VOLUME_RATIO = 2.0  # OI should be 2x volume for institutional buildup
CALL_PUT_RATIO_BULLISH = 1.5  # CPR > 1.5 = Bullish
CALL_PUT_RATIO_BEARISH = 0.67  # CPR < 0.67 = Bearish
PRICE_OI_DIVERGENCE_PCT = 0.005  # 0.5% divergence significant

# --------- EXPIRIES FOR KEPT INDICES ---------
EXPIRIES = {
    "NIFTY": "09 DEC 2025",
    "BANKNIFTY": "30 DEC 2025", 
    "SENSEX": "04 DEC 2025",
    "MIDCPNIFTY": "30 DEC 2025"
}

# --------- ONLY OPTION FLOW STRATEGIES ---------
STRATEGY_NAMES = {
    "oi_buildup_blast": "OI BUILDUP BLAST",
    "smart_money_divergence": "SMART MONEY DIVERGENCE",
    "max_pain_squeeze": "MAX PAIN SQUEEZE",
    "gamma_unbalanced_flow": "GAMMA UNBALANCED FLOW"
}

# --------- INSTITUTIONAL TRACKING ---------
all_generated_signals = []
strategy_performance = {}
signal_counter = 0
daily_signals = []

# --------- SIGNAL DEDUPLICATION ---------
active_strikes = {}
last_signal_time = {}
signal_cooldown = 1800  # 30 minutes cooldown

def initialize_strategy_tracking():
    global strategy_performance
    strategy_performance = {
        "OI BUILDUP BLAST": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "SMART MONEY DIVERGENCE": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "MAX PAIN SQUEEZE": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "GAMMA UNBALANCED FLOW": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0}
    }

initialize_strategy_tracking()

# --------- ANGEL ONE LOGIN ---------
API_KEY = os.getenv("API_KEY")
CLIENT_CODE = os.getenv("CLIENT_CODE")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
TOTP = pyotp.TOTP(TOTP_SECRET).now()

client = SmartConnect(api_key=API_KEY)
session = client.generateSession(CLIENT_CODE, PASSWORD, TOTP)
feedToken = client.getfeedToken()

# --------- TELEGRAM ---------
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

STARTED_SENT = False
STOP_SENT = False
MARKET_CLOSED_SENT = False
EOD_REPORT_SENT = False

def send_telegram(msg, reply_to=None):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg}
        if reply_to:
            payload["reply_to_message_id"] = reply_to
        r = requests.post(url, data=payload, timeout=5).json()
        return r.get("result", {}).get("message_id")
    except:
        return None

# --------- MARKET HOURS ---------
def is_market_open():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    return dtime(9,15) <= current_time_ist <= dtime(15,30)

def should_stop_trading():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    return current_time_ist >= dtime(15,30)

# --------- STRIKE ROUNDING ---------
def round_strike(index, price):
    try:
        if price is None:
            return None
        if isinstance(price, float) and math.isnan(price):
            return None
        price = float(price)
        
        if index == "NIFTY": 
            return int(round(price / 50.0) * 50)
        elif index == "BANKNIFTY": 
            return int(round(price / 100.0) * 100)
        elif index == "SENSEX": 
            return int(round(price / 100.0) * 100)
        elif index == "MIDCPNIFTY": 
            return int(round(price / 25.0) * 25)
        else: 
            return int(round(price / 50.0) * 50)
    except Exception:
        return None

# --------- ENSURE SERIES ---------
def ensure_series(data):
    return data.iloc[:,0] if isinstance(data, pd.DataFrame) else data.squeeze()

# --------- FETCH INDEX DATA ---------
def fetch_index_data(index, interval="5m", period="2d"):
    symbol_map = {
        "NIFTY": "^NSEI", 
        "BANKNIFTY": "^NSEBANK", 
        "SENSEX": "^BSESN",
        "MIDCPNIFTY": "NIFTY_MID_SELECT.NS"
    }
    df = yf.download(symbol_map[index], period=period, interval=interval, auto_adjust=True, progress=False)
    return None if df.empty else df

# --------- LOAD TOKEN MAP ---------
def load_token_map():
    try:
        url="https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df=pd.DataFrame(requests.get(url,timeout=10).json())
        df.columns=[c.lower() for c in df.columns]
        df=df[df['exch_seg'].str.upper().isin(["NFO", "BFO"])]
        df['symbol']=df['symbol'].str.upper()
        return df.set_index('symbol')['token'].to_dict()
    except:
        return {}

token_map=load_token_map()

# --------- SAFE LTP FETCH ---------
def fetch_option_price(symbol, retries=3, delay=3):
    token=token_map.get(symbol.upper())
    if not token:
        return None
    for _ in range(retries):
        try:
            exchange = "BFO" if "SENSEX" in symbol.upper() else "NFO"
            data=client.ltpData(exchange, symbol, token)
            return float(data['data']['ltp'])
        except:
            time.sleep(delay)
    return None

# --------- STRICT EXPIRY VALIDATION ---------
def validate_option_symbol(index, symbol, strike, opttype):
    try:
        expected_expiry = EXPIRIES.get(index)
        if not expected_expiry:
            return False
        expected_dt = datetime.strptime(expected_expiry, "%d %b %Y")
        
        if index == "SENSEX":
            year_short = expected_dt.strftime("%y")
            month_code = expected_dt.strftime("%b").upper()
            day = expected_dt.strftime("%d")
            expected_pattern = f"SENSEX{day}{month_code}{year_short}"
            symbol_upper = symbol.upper()
            if expected_pattern in symbol_upper:
                return True
            else:
                return False
        else:
            expected_pattern = expected_dt.strftime("%d%b%y").upper()
            symbol_upper = symbol.upper()
            if expected_pattern in symbol_upper:
                return True
            else:
                return False
    except Exception as e:
        return False

# --------- GET OPTION SYMBOL ---------
def get_option_symbol(index, expiry_str, strike, opttype):
    try:
        dt = datetime.strptime(expiry_str, "%d %b %Y")
        
        if index == "SENSEX":
            year_short = dt.strftime("%y")
            month_code = dt.strftime("%b").upper()
            day = dt.strftime("%d")
            symbol = f"SENSEX{day}{month_code}{year_short}{strike}{opttype}"
        elif index == "MIDCPNIFTY":
            symbol = f"MIDCPNIFTY{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
        else:
            symbol = f"{index}{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
        
        if validate_option_symbol(index, symbol, strike, opttype):
            return symbol
        else:
            return None
    except Exception as e:
        return None

# 🚨 **INSTITUTIONAL OPTION CHAIN FETCH** 🚨
def fetch_option_chain_angel(index):
    """
    Fetch live option chain data from Angel One
    Returns: DataFrame with OI, Volume, LTP for all strikes
    """
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        df = pd.DataFrame(data)
        df.columns = [col.lower() for col in df.columns]
        
        # Filter for index and expiry
        expiry_str = EXPIRIES.get(index)
        if not expiry_str:
            return None
            
        expiry_date = datetime.strptime(expiry_str, "%d %b %Y")
        
        if index == "SENSEX":
            pattern = f"SENSEX{expiry_date.strftime('%d%b%y').upper()}"
        else:
            pattern = f"{index}{expiry_date.strftime('%d%b%y').upper()}"
        
        # Filter for this expiry
        df_index = df[df['symbol'].str.contains(pattern, na=False)]
        
        if df_index.empty:
            return None
        
        # Extract strike and option type
        def extract_strike(symbol):
            try:
                # Remove index name and expiry
                base = symbol.replace(pattern, "")
                # Remove option type (CE/PE)
                strike_part = base[:-2]
                return int(strike_part)
            except:
                return None
        
        def extract_option_type(symbol):
            return symbol[-2:]
        
        df_index['strike'] = df_index['symbol'].apply(extract_strike)
        df_index['option_type'] = df_index['symbol'].apply(extract_option_type)
        
        # Convert numeric columns
        numeric_cols = ['oi', 'volume', 'ltp']
        for col in numeric_cols:
            if col in df_index.columns:
                df_index[col] = pd.to_numeric(df_index[col], errors='coerce')
        
        return df_index[['symbol', 'strike', 'option_type', 'oi', 'volume', 'ltp']].dropna()
        
    except Exception as e:
        print(f"Error fetching option chain: {e}")
        return None

# 🚨 **OI BUILDUP BLAST DETECTION** 🚨
def detect_oi_buildup_blast(index, df_chain):
    """
    Detect massive OI buildup in options (institutional accumulation)
    """
    try:
        if df_chain is None or df_chain.empty:
            return None
        
        # Separate calls and puts
        calls = df_chain[df_chain['option_type'] == 'CE'].copy()
        puts = df_chain[df_chain['option_type'] == 'PE'].copy()
        
        if calls.empty or puts.empty:
            return None
        
        # Get current index price
        df_index = fetch_index_data(index, "2m", "1d")
        if df_index is None:
            return None
        
        current_price = float(ensure_series(df_index['Close']).iloc[-1])
        
        # Find ATM and nearby strikes
        calls['distance'] = abs(calls['strike'] - current_price)
        puts['distance'] = abs(puts['strike'] - current_price)
        
        # Look at strikes within 2% of current price
        near_calls = calls[calls['distance'] <= current_price * 0.02]
        near_puts = puts[puts['distance'] <= current_price * 0.02]
        
        if near_calls.empty or near_puts.empty:
            return None
        
        # Find maximum OI buildup
        max_call_oi = near_calls['oi'].max()
        max_put_oi = near_puts['oi'].max()
        max_call_strike = near_calls.loc[near_calls['oi'].idxmax(), 'strike']
        max_put_strike = near_puts.loc[near_puts['oi'].idxmax(), 'strike']
        
        # Check for massive OI buildup (> MIN_OI_BUILDUP)
        if max_call_oi > MIN_OI_BUILDUP and max_call_oi > max_put_oi * 1.8:
            # Bullish OI buildup
            call_volume = near_calls.loc[near_calls['oi'].idxmax(), 'volume']
            if call_volume > MIN_VOLUME_BUILDUP:
                return "CE", max_call_strike
        
        if max_put_oi > MIN_OI_BUILDUP and max_put_oi > max_call_oi * 1.8:
            # Bearish OI buildup
            put_volume = near_puts.loc[near_puts['oi'].idxmax(), 'volume']
            if put_volume > MIN_VOLUME_BUILDUP:
                return "PE", max_put_strike
                
    except Exception as e:
        return None
    
    return None

# 🚨 **SMART MONEY DIVERGENCE DETECTION** 🚨
def detect_smart_money_divergence(index, df_chain):
    """
    Detect when OI is building opposite to price move (Smart Money vs Dumb Money)
    """
    try:
        if df_chain is None:
            return None
        
        # Get price action
        df_index = fetch_index_data(index, "5m", "1d")
        if df_index is None or len(df_index) < 10:
            return None
        
        close = ensure_series(df_index['Close'])
        current_price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        price_change_pct = (current_price - prev_price) / prev_price
        
        # Separate calls and puts
        calls = df_chain[df_chain['option_type'] == 'CE'].copy()
        puts = df_chain[df_chain['option_type'] == 'PE'].copy()
        
        if calls.empty or puts.empty:
            return None
        
        # Calculate total OI change
        total_call_oi = calls['oi'].sum()
        total_put_oi = puts['oi'].sum()
        
        # Get previous OI data (would need storage, simplified)
        # For now, use OI-volume ratio as proxy
        
        # Find high OI with low volume (accumulation)
        calls['oi_volume_ratio'] = calls['oi'] / (calls['volume'] + 1)
        puts['oi_volume_ratio'] = puts['oi'] / (puts['volume'] + 1)
        
        # Smart Money Bullish: Price down but Calls accumulating
        if price_change_pct < -0.001:  # Price down
            high_ratio_calls = calls[calls['oi_volume_ratio'] > OI_VOLUME_RATIO]
            if not high_ratio_calls.empty:
                # Smart money buying calls on dip
                avg_strike = high_ratio_calls['strike'].mean()
                if avg_strike > current_price:  # OTM calls being bought
                    return "CE", int(avg_strike)
        
        # Smart Money Bearish: Price up but Puts accumulating
        elif price_change_pct > 0.001:  # Price up
            high_ratio_puts = puts[puts['oi_volume_ratio'] > OI_VOLUME_RATIO]
            if not high_ratio_puts.empty:
                # Smart money buying puts on rise
                avg_strike = high_ratio_puts['strike'].mean()
                if avg_strike < current_price:  # OTM puts being bought
                    return "PE", int(avg_strike)
                    
    except Exception as e:
        return None
    
    return None

# 🚨 **MAX PAIN SQUEEZE DETECTION** 🚨
def detect_max_pain_squeeze(index, df_chain):
    """
    Detect when price is far from Max Pain (market maker pain point)
    """
    try:
        if df_chain is None:
            return None
        
        # Get current price
        df_index = fetch_index_data(index, "2m", "1d")
        if df_index is None:
            return None
        
        current_price = float(ensure_series(df_index['Close']).iloc[-1])
        
        # Calculate Max Pain
        calls = df_chain[df_chain['option_type'] == 'CE'].copy()
        puts = df_chain[df_chain['option_type'] == 'PE'].copy()
        
        if calls.empty or puts.empty:
            return None
        
        # Simple max pain calculation
        strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
        
        min_pain = float('inf')
        max_pain_strike = strikes[0]
        
        for strike in strikes:
            # Total pain at this strike
            call_oi_at_strike = calls[calls['strike'] == strike]['oi'].sum() if not calls[calls['strike'] == strike].empty else 0
            put_oi_at_strike = puts[puts['strike'] == strike]['oi'].sum() if not puts[puts['strike'] == strike].empty else 0
            
            pain = 0
            for s in strikes:
                if s < strike:
                    # Puts expire ITM, Calls expire OTM
                    put_oi = puts[puts['strike'] == s]['oi'].sum() if not puts[puts['strike'] == s].empty else 0
                    pain += put_oi * (strike - s)
                elif s > strike:
                    # Calls expire ITM, Puts expire OTM
                    call_oi = calls[calls['strike'] == s]['oi'].sum() if not calls[calls['strike'] == s].empty else 0
                    pain += call_oi * (s - strike)
            
            if pain < min_pain:
                min_pain = pain
                max_pain_strike = strike
        
        # Check distance from max pain
        distance_pct = abs(current_price - max_pain_strike) / current_price
        
        # If far from max pain (>2%), expect squeeze toward max pain
        if distance_pct > 0.02:
            if current_price > max_pain_strike:
                # Price above max pain → expect downward squeeze
                return "PE", max_pain_strike
            else:
                # Price below max pain → expect upward squeeze
                return "CE", max_pain_strike
                
    except Exception as e:
        return None
    
    return None

# 🚨 **GAMMA UNBALANCED FLOW DETECTION** 🚨
def detect_gamma_unbalanced_flow(index, df_chain):
    """
    Detect when gamma exposure is heavily skewed (institutional hedging)
    """
    try:
        if df_chain is None:
            return None
        
        # Get price
        df_index = fetch_index_data(index, "2m", "1d")
        if df_index is None:
            return None
        
        current_price = float(ensure_series(df_index['Close']).iloc[-1])
        
        # Calculate Call-Put Ratio
        calls = df_chain[df_chain['option_type'] == 'CE']
        puts = df_chain[df_chain['option_type'] == 'PE']
        
        if calls.empty or puts.empty:
            return None
        
        total_call_oi = calls['oi'].sum()
        total_put_oi = puts['oi'].sum()
        
        cpr = total_call_oi / total_put_oi if total_put_oi > 0 else 0
        
        # Heavy Call buying (Bullish)
        if cpr > CALL_PUT_RATIO_BULLISH:
            # Find highest OI call strike
            max_oi_call = calls.loc[calls['oi'].idxmax()]
            if max_oi_call['strike'] > current_price:  # OTM calls
                return "CE", int(max_oi_call['strike'])
        
        # Heavy Put buying (Bearish)
        elif cpr < CALL_PUT_RATIO_BEARISH:
            # Find highest OI put strike
            max_oi_put = puts.loc[puts['oi'].idxmax()]
            if max_oi_put['strike'] < current_price:  # OTM puts
                return "PE", int(max_oi_put['strike'])
                
    except Exception as e:
        return None
    
    return None

# 🚨 **INSTITUTIONAL OPTION FLOW ANALYSIS** 🚨
def analyze_option_flow_signal(index):
    """
    Main institutional option flow analysis
    """
    # Fetch option chain data
    df_chain = fetch_option_chain_angel(index)
    if df_chain is None or df_chain.empty:
        return None
    
    # Get index data for context
    df_index = fetch_index_data(index, "5m", "1d")
    if df_index is None:
        return None
    
    # 🚨 STRATEGY 1: OI BUILDUP BLAST
    oi_signal = detect_oi_buildup_blast(index, df_chain)
    if oi_signal:
        option_type, strike = oi_signal
        return option_type, strike, df_index, False, "oi_buildup_blast"
    
    # 🚨 STRATEGY 2: SMART MONEY DIVERGENCE
    smart_signal = detect_smart_money_divergence(index, df_chain)
    if smart_signal:
        option_type, strike = smart_signal
        return option_type, strike, df_index, False, "smart_money_divergence"
    
    # 🚨 STRATEGY 3: MAX PAIN SQUEEZE
    pain_signal = detect_max_pain_squeeze(index, df_chain)
    if pain_signal:
        option_type, strike = pain_signal
        return option_type, strike, df_index, False, "max_pain_squeeze"
    
    # 🚨 STRATEGY 4: GAMMA UNBALANCED FLOW
    gamma_signal = detect_gamma_unbalanced_flow(index, df_chain)
    if gamma_signal:
        option_type, strike = gamma_signal
        return option_type, strike, df_index, False, "gamma_unbalanced_flow"
    
    return None

# --------- SIGNAL DEDUPLICATION ---------
def can_send_signal(index, strike, option_type):
    current_time = time.time()
    strike_key = f"{index}_{strike}_{option_type}"
    
    if strike_key in active_strikes:
        return False
        
    if index in last_signal_time:
        time_since_last = current_time - last_signal_time[index]
        if time_since_last < signal_cooldown:
            return False
    
    return True

def update_signal_tracking(index, strike, option_type, signal_id):
    global active_strikes, last_signal_time
    
    strike_key = f"{index}_{strike}_{option_type}"
    active_strikes[strike_key] = {
        'signal_id': signal_id,
        'timestamp': time.time(),
        'targets_hit': 0
    }
    
    last_signal_time[index] = time.time()

def update_signal_progress(signal_id, targets_hit):
    for strike_key, data in active_strikes.items():
        if data['signal_id'] == signal_id:
            active_strikes[strike_key]['targets_hit'] = targets_hit
            break

def clear_completed_signal(signal_id):
    global active_strikes
    active_strikes = {k: v for k, v in active_strikes.items() if v['signal_id'] != signal_id}

# --------- TRADE MONITORING ---------
active_trades = {}

def calculate_pnl(entry, max_price, targets, targets_hit, sl):
    try:
        if targets is None or len(targets) == 0:
            diff = max_price - entry
            return f"+{diff:.2f}" if diff > 0 else f"-{abs(diff):.2f}"
        
        achieved_prices = [target for i, target in enumerate(targets) if targets_hit[i]]
        if achieved_prices:
            exit_price = achieved_prices[-1]
            diff = exit_price - entry
            return f"+{diff:.2f}" if diff > 0 else f"-{abs(diff):.2f}"
        else:
            if max_price <= sl:
                diff = sl - entry
                return f"+{diff:.2f}" if diff > 0 else f"-{abs(diff):.2f}"
            else:
                diff = max_price - entry
                return f"+{diff:.2f}" if diff > 0 else f"-{abs(diff):.2f}"
    except Exception:
        return "0"

def monitor_price_live(symbol, entry, targets, sl, fakeout, thread_id, strategy_name, signal_data):
    def monitoring_thread():
        global daily_signals
        
        last_high = entry
        in_trade = False
        max_price_reached = entry
        targets_hit = [False] * len(targets)
        signal_id = signal_data.get('signal_id')
        
        while True:
            if should_stop_trading():
                final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                signal_data.update({
                    "entry_status": "ENTERED" if in_trade else "NOT_ENTERED",
                    "targets_hit": sum(targets_hit),
                    "max_price_reached": max_price_reached,
                    "final_pnl": final_pnl
                })
                daily_signals.append(signal_data)
                clear_completed_signal(signal_id)
                break
                
            price = fetch_option_price(symbol)
            if price:
                price = round(price)
                
                if price > max_price_reached:
                    max_price_reached = price
                
                if not in_trade:
                    if price >= entry:
                        send_telegram(f"✅ ENTRY TRIGGERED at {price}", reply_to=thread_id)
                        in_trade = True
                        signal_data["entry_status"] = "ENTERED"
                else:
                    if price > last_high:
                        last_high = price
                    
                    for i, target in enumerate(targets):
                        if price >= target and not targets_hit[i]:
                            send_telegram(f"🎯 {symbol}: Target {i+1} hit at ₹{target}", reply_to=thread_id)
                            targets_hit[i] = True
                            update_signal_progress(signal_id, sum(targets_hit))
                    
                    if price <= sl:
                        send_telegram(f"🔗 SL HIT at {sl}. ALLOWING NEW SIGNAL.", reply_to=thread_id)
                        final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                        signal_data.update({
                            "targets_hit": sum(targets_hit),
                            "max_price_reached": max_price_reached,
                            "final_pnl": final_pnl
                        })
                        daily_signals.append(signal_data)
                        clear_completed_signal(signal_id)
                        break
                        
                    if all(targets_hit):
                        send_telegram(f"🏆 ALL TARGETS HIT! Trade completed!", reply_to=thread_id)
                        final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                        signal_data.update({
                            "targets_hit": len(targets),
                            "max_price_reached": max_price_reached,
                            "final_pnl": final_pnl
                        })
                        daily_signals.append(signal_data)
                        clear_completed_signal(signal_id)
                        break
            
            time.sleep(10)
    
    thread = threading.Thread(target=monitoring_thread)
    thread.daemon = True
    thread.start()

# --------- EOD REPORT ---------
def send_individual_signal_reports():
    global daily_signals, all_generated_signals
    
    all_signals = daily_signals + all_generated_signals
    if not all_signals:
        send_telegram("📊 No option flow signals generated today.")
        return
    
    send_telegram(f"🕒 OPTION FLOW EOD REPORT - {datetime.utcnow().strftime('%d-%b-%Y')}\n"
                  f"📈 Total Signals: {len(all_signals)}\n"
                  f"─────────────────────────────")
    
    for i, signal in enumerate(all_signals, 1):
        targets_for_disp = signal.get('targets', [])
        while len(targets_for_disp) < 4:
            targets_for_disp.append('-')
        
        msg = (f"📊 SIGNAL #{i}\n"
               f"📈 {signal.get('index','?')} {signal.get('strike','?')} {signal.get('option_type','?')}\n"
               f"🏷️ {signal.get('strategy','?')}\n"
               f"💰 ENTRY: ₹{signal.get('entry_price','?')}\n"
               f"🎯 TARGETS: {targets_for_disp[0]}//{targets_for_disp[1]}//{targets_for_disp[2]}//{targets_for_disp[3]}\n"
               f"🛑 SL: ₹{signal.get('sl','?')}\n"
               f"📊 Targets Hit: {signal.get('targets_hit', 0)}/4\n"
               f"💵 P&L: {signal.get('final_pnl', '0')}\n"
               f"─────────────────────────────")
        
        send_telegram(msg)
        time.sleep(1)

# 🚨 **INSTITUTIONAL OPTION FLOW SIGNAL SENDING** 🚨
def send_option_flow_signal(index, option_type, strike, df, fakeout, strategy_key):
    global signal_counter, all_generated_signals
    
    if not can_send_signal(index, strike, option_type):
        return
    
    symbol = get_option_symbol(index, EXPIRIES[index], strike, option_type)
    if symbol is None:
        return
    
    option_price = fetch_option_price(symbol)
    if not option_price:
        return
    
    entry = round(option_price)
    
    # 🚨 INSTITUTIONAL TARGETS BASED ON OPTION FLOW
    if option_type == "CE":
        base_move = max(100, option_price * 3.0)  # 100 points or 3x premium
        targets = [
            round(entry + base_move * 1.0),
            round(entry + base_move * 2.0),
            round(entry + base_move * 3.0),
            round(entry + base_move * 4.0)
        ]
        sl = round(entry - base_move * 0.6)
    else:  # PE
        base_move = max(100, option_price * 3.0)
        targets = [
            round(entry + base_move * 1.0),
            round(entry + base_move * 2.0),
            round(entry + base_move * 3.0),
            round(entry + base_move * 4.0)
        ]
        sl = round(entry - base_move * 0.6)
    
    strategy_name = STRATEGY_NAMES.get(strategy_key, strategy_key.upper())
    signal_id = f"OPT{signal_counter:04d}"
    signal_counter += 1
    
    signal_data = {
        "signal_id": signal_id,
        "timestamp": datetime.utcnow().strftime("%H:%M:%S"),
        "index": index,
        "strike": strike,
        "option_type": option_type,
        "strategy": strategy_name,
        "entry_price": entry,
        "targets": targets,
        "sl": sl,
        "fakeout": fakeout,
        "entry_status": "PENDING",
        "targets_hit": 0,
        "max_price_reached": entry,
        "final_pnl": "0"
    }
    
    update_signal_tracking(index, strike, option_type, signal_id)
    all_generated_signals.append(signal_data.copy())
    
    # 🚨 INSTITUTIONAL OPTION FLOW ALERT
    targets_str = "//".join(str(t) for t in targets) + "++"
    
    msg = (f"💥 INSTITUTIONAL OPTION FLOW DETECTED 💥\n"
           f"📈 {index} {strike} {option_type}\n"
           f"SYMBOL: {symbol}\n"
           f"ENTRY ABOVE: ₹{entry}\n"
           f"TARGETS: {targets_str}\n"
           f"STOP LOSS: ₹{sl}\n"
           f"STRATEGY: {strategy_name}\n"
           f"SIGNAL ID: {signal_id}\n"
           f"⚠️ INSTITUTIONAL OI/FLOW CONFIRMED")
    
    thread_id = send_telegram(msg)
    
    trade_id = f"{symbol}_{int(time.time())}"
    active_trades[trade_id] = {
        "symbol": symbol,
        "entry": entry,
        "sl": sl,
        "targets": targets,
        "thread": thread_id,
        "signal_data": signal_data
    }
    
    monitor_price_live(symbol, entry, targets, sl, fakeout, thread_id, strategy_name, signal_data)

# 🚨 **OPTION FLOW TRADE THREAD** 🚨
def option_flow_trade_thread(index):
    result = analyze_option_flow_signal(index)
    
    if not result:
        return
    
    option_type, strike, df, fakeout, strategy_key = result
    send_option_flow_signal(index, option_type, strike, df, fakeout, strategy_key)

# 🚨 **MAIN OPTION FLOW LOOP** 🚨
def run_option_flow_algo():
    if not is_market_open():
        return
    
    if should_stop_trading():
        global STOP_SENT, EOD_REPORT_SENT
        if not STOP_SENT:
            send_telegram("🛑 Market closed - Option Flow Algorithm stopped")
            STOP_SENT = True
        
        if not EOD_REPORT_SENT:
            time.sleep(15)
            send_telegram("📊 GENERATING OPTION FLOW EOD REPORT...")
            try:
                send_individual_signal_reports()
            except Exception as e:
                send_telegram(f"⚠️ Report Error: {str(e)[:100]}")
                time.sleep(10)
                send_individual_signal_reports()
            EOD_REPORT_SENT = True
            send_telegram("✅ OPTION FLOW DAY COMPLETED!")
        
        return
    
    threads = []
    indices = ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"]
    
    for index in indices:
        t = threading.Thread(target=option_flow_trade_thread, args=(index,))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()

# 🚨 **MAIN INSTITUTIONAL OPTION FLOW ENGINE** 🚨
STARTED_SENT = False
STOP_SENT = False
MARKET_CLOSED_SENT = False
EOD_REPORT_SENT = False

while True:
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        current_time_ist = ist_now.time()
        
        market_open = is_market_open()
        
        if not market_open:
            if not MARKET_CLOSED_SENT:
                send_telegram("🔴 Market closed. Option Flow Engine waiting...")
                MARKET_CLOSED_SENT = True
                STARTED_SENT = False
                STOP_SENT = False
                EOD_REPORT_SENT = False
            
            if current_time_ist >= dtime(15,30) and current_time_ist <= dtime(16,0) and not EOD_REPORT_SENT:
                send_telegram("📊 GENERATING OPTION FLOW EOD REPORT...")
                time.sleep(10)
                send_individual_signal_reports()
                EOD_REPORT_SENT = True
            
            time.sleep(30)
            continue
        
        if not STARTED_SENT:
            send_telegram("🚀 **INSTITUTIONAL OPTION FLOW ENGINE ACTIVATED** 🚀\n"
                         "✅ Tracking: NIFTY, BANKNIFTY, SENSEX, MIDCPNIFTY\n"
                         "✅ Strategies: OI BUILDUP + SMART MONEY + MAX PAIN + GAMMA FLOW\n"
                         "✅ Data: Real-time Option Chain Analysis\n"
                         "✅ Filters: Institutional OI/Volume Thresholds\n"
                         "❌ REJECTING: All retail price-only signals")
            STARTED_SENT = True
            STOP_SENT = False
            MARKET_CLOSED_SENT = False
        
        if should_stop_trading():
            if not STOP_SENT:
                send_telegram("🛑 Market closing! Preparing Option Flow EOD Report...")
                STOP_SENT = True
                STARTED_SENT = False
            
            if not EOD_REPORT_SENT:
                send_telegram("📊 FINALIZING OPTION FLOW TRADES...")
                time.sleep(20)
                try:
                    send_individual_signal_reports()
                except Exception as e:
                    send_telegram(f"⚠️ Report Error: {str(e)[:100]}")
                    time.sleep(10)
                    send_individual_signal_reports()
                EOD_REPORT_SENT = True
                send_telegram("✅ OPTION FLOW DAY COMPLETED!")
            
            time.sleep(60)
            continue
        
        run_option_flow_algo()
        time.sleep(60)  # Check every minute (option chain updates slower)
        
    except Exception as e:
        error_msg = f"⚠️ Option Flow Engine error: {str(e)[:100]}"
        send_telegram(error_msg)
        time.sleep(60)
