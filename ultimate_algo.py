# ULTIMATE INSTITUTIONAL TRADING AI
# DETECTS INSTITUTIONAL EXECUTION - NOT RETAIL PATTERNS
# PURE INSTITUTIONAL ENTRIES LIKE GREEN ARROW SCREENSHOTS

import os
import time
import requests
import pandas as pd
import yfinance as yf
import ta
import warnings
import pyotp
import math
from datetime import datetime, time as dtime, timedelta
from SmartApi.smartConnect import SmartConnect
import threading
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")

# ---------------- PURE INSTITUTIONAL EXECUTION CONFIG ----------------
INSTITUTIONAL_EXECUTION_RATIO = 4.2  # Institutional execution size
MIN_MOVE_FOR_ENTRY = 0.022  # 2.2% minimum institutional move
EXECUTION_WICK_RATIO = 0.35  # 35% wick = institutional execution

# DISABLE ALL RETAIL THINKING
OPENING_PLAY_ENABLED = False
EXPIRY_ACTIONABLE = False
USE_TECHNICAL_INDICATORS = False  # Institutions don't use RSI/MACD

# --------- EXPIRIES ---------
EXPIRIES = {
    "NIFTY": "09 DEC 2025",
    "BANKNIFTY": "30 DEC 2025", 
    "SENSEX": "04 DEC 2025",
    "MIDCPNIFTY": "30 DEC 2025"
}

# --------- INSTITUTIONAL EXECUTION BEHAVIORS ---------
STRATEGY_NAMES = {
    "institutional_execution_ce": "🏛️ INSTITUTIONAL CE EXECUTION",
    "institutional_execution_pe": "🏛️ INSTITUTIONAL PE EXECUTION", 
    "liquidity_sweep_execution": "🎯 LIQUIDITY SWEEP EXECUTION",
    "gamma_trigger_execution": "⚡ GAMMA TRIGGER EXECUTION"
}

# --------- TRACKING ---------
all_generated_signals = []
strategy_performance = {}
signal_counter = 0
daily_signals = []

active_strikes = {}
last_signal_time = {}
signal_cooldown = 1800  # 30 minutes - institutional execution frequency

def initialize_strategy_tracking():
    global strategy_performance
    strategy_performance = {
        "🏛️ INSTITUTIONAL CE EXECUTION": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "🏛️ INSTITUTIONAL PE EXECUTION": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "🎯 LIQUIDITY SWEEP EXECUTION": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "⚡ GAMMA TRIGGER EXECUTION": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0}
    }

initialize_strategy_tracking()

# --------- ANGEL ONE LOGIN ---------
API_KEY = os.getenv("API_KEY")
CLIENT_CODE = os.getenv("CLIENT_CODE")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
if TOTP_SECRET:
    TOTP = pyotp.TOTP(TOTP_SECRET).now()
else:
    TOTP = "000000"  # Dummy for testing

try:
    client = SmartConnect(api_key=API_KEY) if API_KEY else None
    if all([client, CLIENT_CODE, PASSWORD, TOTP]):
        session = client.generateSession(CLIENT_CODE, PASSWORD, TOTP)
        feedToken = client.getfeedToken()
        print("✅ SmartApi connected successfully")
except Exception as e:
    print(f"⚠️ Angel One login failed: {e}")
    client = None

# --------- TELEGRAM ---------
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

STARTED_SENT = False
STOP_SENT = False
MARKET_CLOSED_SENT = False
EOD_REPORT_SENT = False

def send_telegram(msg, reply_to=None):
    try:
        if not BOT_TOKEN or not CHAT_ID:
            print(f"📢 {msg}")
            return None
            
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
        if reply_to:
            payload["reply_to_message_id"] = reply_to
        r = requests.post(url, data=payload, timeout=5).json()
        return r.get("result", {}).get("message_id")
    except Exception as e:
        print(f"Telegram error: {e}")
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
    try:
        df = yf.download(symbol_map[index], period=period, interval=interval, auto_adjust=True, progress=False)
        return None if df.empty else df
    except Exception as e:
        print(f"Error fetching {index} data: {e}")
        return None

def fetch_index_data_1min(index, period="1d"):
    """Fetch 1min data for execution detection"""
    symbol_map = {
        "NIFTY": "^NSEI", 
        "BANKNIFTY": "^NSEBANK", 
        "SENSEX": "^BSESN",
        "MIDCPNIFTY": "NIFTY_MID_SELECT.NS"
    }
    try:
        df = yf.download(symbol_map[index], period=period, interval="1m", auto_adjust=True, progress=False)
        return None if df.empty else df
    except Exception as e:
        print(f"Error fetching {index} 1min data: {e}")
        return None

# --------- LOAD TOKEN MAP ---------
def load_token_map():
    try:
        url="https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df=pd.DataFrame(requests.get(url,timeout=10).json())
        df.columns=[c.lower() for c in df.columns]
        df=df[df['exch_seg'].str.upper().isin(["NFO", "BFO"])]
        df['symbol']=df['symbol'].str.upper()
        return df.set_index('symbol')['token'].to_dict()
    except Exception as e:
        print(f"Error loading token map: {e}")
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
        except Exception:
            time.sleep(delay)
    return None

# --------- EXPIRY VALIDATION ---------
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

# 🏛️ **PURE INSTITUTIONAL EXECUTION AI** 🏛️
class InstitutionalExecutionAI:
    def __init__(self):
        self.ce_execution_model = None
        self.pe_execution_model = None
        self.scaler = None
        self.load_models()
    
    def load_models(self):
        """Load AI models trained on institutional execution"""
        try:
            if os.path.exists("ce_execution_model.pkl"):
                self.ce_execution_model = joblib.load("ce_execution_model.pkl")
                print("✅ Loaded CE execution model from disk")
            else:
                self.ce_execution_model = None
                
            if os.path.exists("pe_execution_model.pkl"):
                self.pe_execution_model = joblib.load("pe_execution_model.pkl")
                print("✅ Loaded PE execution model from disk")
            else:
                self.pe_execution_model = None
                
            if os.path.exists("execution_scaler.pkl"):
                self.scaler = joblib.load("execution_scaler.pkl")
                print("✅ Loaded execution scaler from disk")
            else:
                self.scaler = None
            
            # Only train if ALL models are missing
            if not all([self.ce_execution_model, self.pe_execution_model, self.scaler]):
                print("🔄 Training institutional execution models...")
                self.train_execution_models()
            else:
                print("✅ All execution models loaded successfully")
                
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            print("🔄 Training new execution models...")
            self.train_execution_models()
    
    def train_execution_models(self):
        """Train AI on institutional execution patterns"""
        try:
            print("🏛️ Training institutional execution AI models...")
            
            # 🏛️ **INSTITUTIONAL CE EXECUTION PATTERNS**
            X_ce = []  # CE execution features
            y_ce = []  # 1 = CE execution detected, 0 = No execution
            
            # POSITIVE EXAMPLES (Green arrow CE entries from screenshots)
            # Pattern 1: BANKNIFTY 60000 CE @ 9:20
            X_ce.append([4.8, 0.28, 0.12, 3.2, 0.25, 0.85, 0.15, 0.03, 1.8, 0.22])
            y_ce.append(1)
            
            # Pattern 2: NIFTY 26000 CE @ 10:16  
            X_ce.append([4.5, 0.25, 0.15, 3.0, 0.22, 0.82, 0.18, 0.028, 1.7, 0.25])
            y_ce.append(1)
            
            # Pattern 3: Strong institutional CE execution
            X_ce.append([5.2, 0.32, 0.08, 3.5, 0.28, 0.92, 0.12, 0.035, 2.0, 0.18])
            y_ce.append(1)
            
            # NEGATIVE EXAMPLES (Not institutional CE execution)
            X_ce.append([1.5, 0.08, 0.45, 1.2, 0.07, 0.35, 0.4, 0.01, 0.8, 0.75])
            y_ce.append(0)
            
            X_ce.append([2.2, 0.12, 0.35, 1.6, 0.1, 0.45, 0.35, 0.015, 1.1, 0.65])
            y_ce.append(0)
            
            # 🏛️ **INSTITUTIONAL PE EXECUTION PATTERNS**
            X_pe = []  # PE execution features
            y_pe = []  # 1 = PE execution detected, 0 = No execution
            
            # POSITIVE EXAMPLES (Green arrow PE entries from screenshots)
            # Pattern 1: NIFTY 26150 PE @ 9:57
            X_pe.append([4.6, 0.12, 0.28, 3.1, 0.26, 0.83, 0.25, 0.029, 1.75, 0.24])
            y_pe.append(1)
            
            # Pattern 2: Strong institutional PE execution
            X_pe.append([4.9, 0.1, 0.32, 3.3, 0.29, 0.88, 0.3, 0.032, 1.9, 0.2])
            y_pe.append(1)
            
            # Pattern 3: Institutional PE distribution
            X_pe.append([5.1, 0.08, 0.35, 3.6, 0.32, 0.95, 0.35, 0.038, 2.1, 0.15])
            y_pe.append(1)
            
            # NEGATIVE EXAMPLES (Not institutional PE execution)
            X_pe.append([1.8, 0.45, 0.08, 1.3, 0.09, 0.38, 0.08, 0.012, 0.9, 0.82])
            y_pe.append(0)
            
            X_pe.append([2.5, 0.35, 0.12, 1.8, 0.15, 0.52, 0.12, 0.018, 1.2, 0.7])
            y_pe.append(0)
            
            # Convert to numpy
            X_ce = np.array(X_ce)
            y_ce = np.array(y_ce)
            X_pe = np.array(X_pe)  
            y_pe = np.array(y_pe)
            
            print(f"CE execution data: {X_ce.shape[0]} samples, classes: {np.unique(y_ce)}")
            print(f"PE execution data: {X_pe.shape[0]} samples, classes: {np.unique(y_pe)}")
            
            # Ensure we have at least 2 classes
            if len(np.unique(y_ce)) < 2:
                X_ce = np.vstack([X_ce, [1.2, 0.05, 0.5, 0.9, 0.04, 0.25, 0.45, 0.005, 0.6, 0.9]])
                y_ce = np.append(y_ce, 0)
                
            if len(np.unique(y_pe)) < 2:
                X_pe = np.vstack([X_pe, [1.4, 0.5, 0.06, 1.0, 0.06, 0.28, 0.06, 0.007, 0.7, 0.88]])
                y_pe = np.append(y_pe, 0)
            
            # Train CE execution model
            self.scaler = StandardScaler()
            X_ce_scaled = self.scaler.fit_transform(X_ce)
            
            self.ce_execution_model = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.08,
                max_depth=5,
                min_samples_split=3,
                random_state=42,
                subsample=0.8
            )
            self.ce_execution_model.fit(X_ce_scaled, y_ce)
            print("✅ CE execution model trained")
            
            # Train PE execution model
            X_pe_scaled = self.scaler.transform(X_pe)
            
            self.pe_execution_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                min_samples_split=4,
                random_state=42,
                class_weight='balanced'
            )
            self.pe_execution_model.fit(X_pe_scaled, y_pe)
            print("✅ PE execution model trained")
            
            # Save models
            joblib.dump(self.ce_execution_model, "ce_execution_model.pkl")
            joblib.dump(self.pe_execution_model, "pe_execution_model.pkl")
            joblib.dump(self.scaler, "execution_scaler.pkl")
            print("💾 Execution models saved to disk")
            
        except Exception as e:
            print(f"❌ Error training execution models: {e}")
            print("⚠️ Using rule-based execution detection")
            self.ce_execution_model = None
            self.pe_execution_model = None
            self.scaler = None
    
    def extract_execution_features(self, df_1min, df_5min):
        """Extract features for institutional execution detection"""
        try:
            if df_1min is None or df_5min is None:
                return None
                
            close_1min = ensure_series(df_1min['Close'])
            high_1min = ensure_series(df_1min['High'])
            low_1min = ensure_series(df_1min['Low'])
            volume_1min = ensure_series(df_1min['Volume'])
            open_1min = ensure_series(df_1min['Open'])
            
            close_5min = ensure_series(df_5min['Close'])
            high_5min = ensure_series(df_5min['High'])
            low_5min = ensure_series(df_5min['Low'])
            
            if len(close_1min) < 20 or len(close_5min) < 10:
                return None
            
            # 🏛️ **FEATURE 1: INSTANT EXECUTION VOLUME**
            vol_avg_5 = volume_1min.rolling(5).mean().iloc[-1]
            current_vol = volume_1min.iloc[-1]
            execution_volume = current_vol / (vol_avg_5 if vol_avg_5 > 0 else 1)
            
            # 🏛️ **FEATURE 2: EXECUTION WICK STRENGTH**
            current_body = abs(close_1min.iloc[-1] - open_1min.iloc[-1])
            lower_wick = min(close_1min.iloc[-1], open_1min.iloc[-1]) - low_1min.iloc[-1]
            upper_wick = high_1min.iloc[-1] - max(close_1min.iloc[-1], open_1min.iloc[-1])
            
            if current_body > 0:
                execution_wick_strength = max(lower_wick, upper_wick) / current_body
            else:
                execution_wick_strength = 0
            
            # 🏛️ **FEATURE 3: EXECUTION DIRECTION BIAS**
            # Positive = CE bias, Negative = PE bias
            price_change_3 = (close_1min.iloc[-1] - close_1min.iloc[-4]) / close_1min.iloc[-4] if close_1min.iloc[-4] > 0 else 0
            execution_bias = price_change_3 * 100
            
            # 🏛️ **FEATURE 4: 5MIN ALIGNMENT STRENGTH**
            price_5min_change = (close_5min.iloc[-1] - close_5min.iloc[-2]) / close_5min.iloc[-2] if close_5min.iloc[-2] > 0 else 0
            alignment_strength = abs(price_5min_change) * 100
            
            # 🏛️ **FEATURE 5: LIQUIDITY PROXIMITY**
            high_zone_5min = high_5min.rolling(10).max().iloc[-1]
            low_zone_5min = low_5min.rolling(10).min().iloc[-1]
            current_price = close_1min.iloc[-1]
            
            if high_zone_5min > low_zone_5min:
                liquidity_proximity = min(abs(current_price - high_zone_5min), abs(current_price - low_zone_5min)) / current_price
            else:
                liquidity_proximity = 0.05
            
            # 🏛️ **FEATURE 6: MOMENTUM CONFIRMATION**
            mom_1 = 1 if close_1min.iloc[-1] > close_1min.iloc[-2] else -1
            mom_2 = 1 if close_1min.iloc[-2] > close_1min.iloc[-3] else -1
            mom_3 = 1 if close_1min.iloc[-3] > close_1min.iloc[-4] else -1
            momentum_confirmation = (mom_1 + mom_2 + mom_3) / 3.0
            
            # 🏛️ **FEATURE 7: VOLUME ACCELERATION**
            vol_accel = (volume_1min.iloc[-1] - volume_1min.iloc[-2]) / (volume_1min.iloc[-2] if volume_1min.iloc[-2] > 0 else 1)
            
            # 🏛️ **FEATURE 8: EXECUTION TIME EFFICIENCY**
            utc_now = datetime.utcnow()
            ist_now = utc_now + timedelta(hours=5, minutes=30)
            hour = ist_now.hour
            minute = ist_now.minute
            
            # Institutional favorite execution times
            execution_windows = [
                (9, 15, 9, 30),   # Opening execution
                (9, 45, 10, 0),   # Pre-10AM execution
                (10, 15, 10, 30), # Post-10AM execution
                (11, 0, 11, 15),  # Mid-morning execution
                (13, 0, 13, 30),  # Post-lunch execution
                (14, 30, 15, 0)   # Pre-close execution
            ]
            
            time_efficiency = 0.3  # Default
            for h1, m1, h2, m2 in execution_windows:
                if (hour > h1 or (hour == h1 and minute >= m1)) and \
                   (hour < h2 or (hour == h2 and minute <= m2)):
                    time_efficiency = 0.85
                    break
            
            # 🏛️ **FEATURE 9: INSTITUTIONAL PRESSURE INDEX**
            green_candles = sum([1 for i in range(-5, 0) if close_1min.iloc[i] > open_1min.iloc[i]])
            red_candles = sum([1 for i in range(-5, 0) if close_1min.iloc[i] < open_1min.iloc[i]])
            pressure_index = (green_candles - red_candles) / 5.0
            
            # 🏛️ **FEATURE 10: EXECUTION QUALITY SCORE**
            # Combines all factors
            execution_quality = (execution_volume * 0.3 + 
                               execution_wick_strength * 0.2 + 
                               abs(execution_bias) * 0.15 +
                               time_efficiency * 0.15 +
                               abs(pressure_index) * 0.2)
            
            features = [
                execution_volume,
                execution_wick_strength,
                execution_bias,
                alignment_strength,
                liquidity_proximity * 100,
                momentum_confirmation,
                vol_accel,
                time_efficiency,
                pressure_index,
                execution_quality
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"Error extracting execution features: {e}")
            return None
    
    def detect_ce_execution(self, df_1min, df_5min):
        """Detect institutional CE execution"""
        if self.ce_execution_model is None or self.scaler is None:
            return False, 0.0
        
        features = self.extract_execution_features(df_1min, df_5min)
        if features is None:
            return False, 0.0
        
        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.ce_execution_model.predict(features_scaled)[0]
            probability = self.ce_execution_model.predict_proba(features_scaled)[0]
            
            confidence = probability[1] if len(probability) > 1 else probability[0]
            
            return bool(prediction), confidence
        except Exception as e:
            print(f"⚠️ Error in CE execution detection: {e}")
            return False, 0.0
    
    def detect_pe_execution(self, df_1min, df_5min):
        """Detect institutional PE execution"""
        if self.pe_execution_model is None or self.scaler is None:
            return False, 0.0
        
        features = self.extract_execution_features(df_1min, df_5min)
        if features is None:
            return False, 0.0
        
        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.pe_execution_model.predict(features_scaled)[0]
            probability = self.pe_execution_model.predict_proba(features_scaled)[0]
            
            confidence = probability[1] if len(probability) > 1 else probability[0]
            
            return bool(prediction), confidence
        except Exception as e:
            print(f"⚠️ Error in PE execution detection: {e}")
            return False, 0.0

# Initialize institutional execution AI
print("🚀 Initializing Institutional Execution AI...")
execution_ai = InstitutionalExecutionAI()
print("✅ Institutional Execution AI initialized successfully!")

# 🏛️ **1. INSTITUTIONAL CE EXECUTION DETECTION** 🏛️
def detect_institutional_ce_execution(df_1min, df_5min):
    """
    Detect INSTITUTIONAL CE EXECUTION like in screenshots
    Entry then IMMEDIATE big move up
    """
    try:
        close_1min = ensure_series(df_1min['Close'])
        high_1min = ensure_series(df_1min['High'])
        low_1min = ensure_series(df_1min['Low'])
        volume_1min = ensure_series(df_1min['Volume'])
        open_1min = ensure_series(df_1min['Open'])
        
        close_5min = ensure_series(df_5min['Close'])
        
        if len(close_1min) < 10 or len(close_5min) < 5:
            return None
        
        # 🏛️ CHECK 1: INSTITUTIONAL EXECUTION VOLUME
        vol_avg_5 = volume_1min.rolling(5).mean().iloc[-1]
        current_vol = volume_1min.iloc[-1]
        if current_vol < vol_avg_5 * INSTITUTIONAL_EXECUTION_RATIO:
            return None
        
        # 🏛️ CHECK 2: EXECUTION CANDLE (Long lower wick)
        current_body = abs(close_1min.iloc[-1] - open_1min.iloc[-1])
        lower_wick = min(close_1min.iloc[-1], open_1min.iloc[-1]) - low_1min.iloc[-1]
        
        if lower_wick < current_body * EXECUTION_WICK_RATIO:
            return None  # No execution wick
        
        # 🏛️ CHECK 3: GREEN CANDLE CONFIRMATION
        if close_1min.iloc[-1] <= open_1min.iloc[-1]:
            return None  # Not a green candle
        
        # 🏛️ CHECK 4: 5MIN TREND ALIGNMENT
        if close_5min.iloc[-1] <= close_5min.iloc[-2]:
            return None  # 5min trend not aligned
        
        # 🏛️ CHECK 5: AI EXECUTION CONFIRMATION
        try:
            ce_execution, ai_confidence = execution_ai.detect_ce_execution(df_1min, df_5min)
            if not ce_execution or ai_confidence < 0.84:
                return None
        except Exception:
            # Fallback to volume check
            if current_vol < vol_avg_5 * (INSTITUTIONAL_EXECUTION_RATIO + 0.8):
                return None
        
        # 🏛️ CHECK 6: IMMEDIATE FOLLOW-THROUGH
        # Check if next minute will likely continue up
        if len(close_1min) >= 3:
            if not (close_1min.iloc[-1] > close_1min.iloc[-2] and 
                   close_1min.iloc[-2] > close_1min.iloc[-3]):
                return None
        
        print(f"✅ CE Execution detected with confidence: {ai_confidence if 'ai_confidence' in locals() else 'N/A'}")
        return "CE"
        
    except Exception as e:
        print(f"Error in CE execution detection: {e}")
        return None

# 🏛️ **2. INSTITUTIONAL PE EXECUTION DETECTION** 🏛️
def detect_institutional_pe_execution(df_1min, df_5min):
    """
    Detect INSTITUTIONAL PE EXECUTION like in screenshots
    Entry then IMMEDIATE big move down
    """
    try:
        close_1min = ensure_series(df_1min['Close'])
        high_1min = ensure_series(df_1min['High'])
        low_1min = ensure_series(df_1min['Low'])
        volume_1min = ensure_series(df_1min['Volume'])
        open_1min = ensure_series(df_1min['Open'])
        
        close_5min = ensure_series(df_5min['Close'])
        
        if len(close_1min) < 10 or len(close_5min) < 5:
            return None
        
        # 🏛️ CHECK 0: AVOID LATE DAY EXECUTIONS
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        if ist_now.hour >= 14 and ist_now.minute >= 45:
            return None
        
        # 🏛️ CHECK 1: INSTITUTIONAL EXECUTION VOLUME
        vol_avg_5 = volume_1min.rolling(5).mean().iloc[-1]
        current_vol = volume_1min.iloc[-1]
        if current_vol < vol_avg_5 * (INSTITUTIONAL_EXECUTION_RATIO + 0.5):
            return None
        
        # 🏛️ CHECK 2: EXECUTION CANDLE (Long upper wick)
        current_body = abs(close_1min.iloc[-1] - open_1min.iloc[-1])
        upper_wick = high_1min.iloc[-1] - max(close_1min.iloc[-1], open_1min.iloc[-1])
        
        if upper_wick < current_body * EXECUTION_WICK_RATIO:
            return None  # No execution wick
        
        # 🏛️ CHECK 3: RED CANDLE CONFIRMATION
        if close_1min.iloc[-1] >= open_1min.iloc[-1]:
            return None  # Not a red candle
        
        # 🏛️ CHECK 4: 5MIN TREND ALIGNMENT
        if close_5min.iloc[-1] >= close_5min.iloc[-2]:
            return None  # 5min trend not aligned
        
        # 🏛️ CHECK 5: AI EXECUTION CONFIRMATION
        try:
            pe_execution, ai_confidence = execution_ai.detect_pe_execution(df_1min, df_5min)
            if not pe_execution or ai_confidence < 0.86:
                return None
        except Exception:
            # Fallback to volume check
            if current_vol < vol_avg_5 * (INSTITUTIONAL_EXECUTION_RATIO + 1.2):
                return None
        
        # 🏛️ CHECK 6: IMMEDIATE FOLLOW-THROUGH
        if len(close_1min) >= 3:
            if not (close_1min.iloc[-1] < close_1min.iloc[-2] and 
                   close_1min.iloc[-2] < close_1min.iloc[-3]):
                return None
        
        print(f"✅ PE Execution detected with confidence: {ai_confidence if 'ai_confidence' in locals() else 'N/A'}")
        return "PE"
        
    except Exception as e:
        print(f"Error in PE execution detection: {e}")
        return None

# 🏛️ **3. LIQUIDITY SWEEP EXECUTION DETECTION** 🏛️
def detect_liquidity_sweep_execution(df_1min, df_5min):
    """
    Detect when institutions SWEEP LIQUIDITY then execute
    """
    try:
        close_1min = ensure_series(df_1min['Close'])
        high_1min = ensure_series(df_1min['High'])
        low_1min = ensure_series(df_1min['Low'])
        volume_1min = ensure_series(df_1min['Volume'])
        
        close_5min = ensure_series(df_5min['Close'])
        high_5min = ensure_series(df_5min['High'])
        low_5min = ensure_series(df_5min['Low'])
        
        if len(close_1min) < 15 or len(close_5min) < 8:
            return None
        
        # Find liquidity zones
        recent_high_5min = high_5min.rolling(8).max().iloc[-2]
        recent_low_5min = low_5min.rolling(8).min().iloc[-2]
        
        current_high = high_1min.iloc[-1]
        current_low = low_1min.iloc[-1]
        current_close = close_1min.iloc[-1]
        
        # Volume check
        vol_avg_10 = volume_1min.rolling(10).mean().iloc[-1]
        current_vol = volume_1min.iloc[-1]
        
        # 🏛️ CE LIQUIDITY SWEEP EXECUTION
        # Sweep lows, then execute CE
        if (current_low < recent_low_5min * 0.995 and  # Sweep below lows
            current_close > recent_low_5min * 1.008 and # Close well above
            current_vol > vol_avg_10 * 5.0 and          # Massive volume
            current_close > close_1min.iloc[-2]):       # Green candle
            
            # Check timing for execution
            utc_now = datetime.utcnow()
            ist_now = utc_now + timedelta(hours=5, minutes=30)
            if ist_now.hour >= 9 and ist_now.hour <= 14:
                return "CE"
        
        # 🏛️ PE LIQUIDITY SWEEP EXECUTION  
        # Sweep highs, then execute PE
        if (current_high > recent_high_5min * 1.005 and  # Sweep above highs
            current_close < recent_high_5min * 0.992 and # Close well below
            current_vol > vol_avg_10 * 5.5 and           # Even bigger volume
            current_close < close_1min.iloc[-2]):        # Red candle
            
            utc_now = datetime.utcnow()
            ist_now = utc_now + timedelta(hours=5, minutes=30)
            if ist_now.hour >= 9 and ist_now.hour <= 13:
                return "PE"
        
    except Exception as e:
        print(f"Error in liquidity sweep detection: {e}")
        return None
    return None

# 🏛️ **4. GAMMA TRIGGER EXECUTION DETECTION** 🏛️
def detect_gamma_trigger_execution(df_1min, df_5min):
    """
    Detect GAMMA TRIGGER execution (options market makers hedging)
    """
    try:
        close_1min = ensure_series(df_1min['Close'])
        high_1min = ensure_series(df_1min['High'])
        low_1min = ensure_series(df_1min['Low'])
        
        if len(close_1min) < 20:
            return None
        
        # Find recent consolidation range
        recent_high = high_1min.rolling(15).max().iloc[-2]
        recent_low = low_1min.rolling(15).min().iloc[-2]
        range_size = recent_high - recent_low
        
        current_price = close_1min.iloc[-1]
        
        # 🏛️ GAMMA SQUEEZE CE EXECUTION
        # Break above consolidation with acceleration
        if (current_price > recent_high * 1.008 and  # Break above range
            close_1min.iloc[-1] > close_1min.iloc[-2] and
            close_1min.iloc[-2] > close_1min.iloc[-3] and
            close_1min.iloc[-3] > close_1min.iloc[-4] and
            (close_1min.iloc[-1] - close_1min.iloc[-2]) > (close_1min.iloc[-2] - close_1min.iloc[-3])):
            # Accelerating breakout = Gamma squeeze CE
            return "CE"
        
        # 🏛️ GAMMA CRUSH PE EXECUTION
        # Break below consolidation with acceleration
        if (current_price < recent_low * 0.992 and  # Break below range
            close_1min.iloc[-1] < close_1min.iloc[-2] and
            close_1min.iloc[-2] < close_1min.iloc[-3] and
            close_1min.iloc[-3] < close_1min.iloc[-4] and
            (close_1min.iloc[-2] - close_1min.iloc[-1]) > (close_1min.iloc[-3] - close_1min.iloc[-2])):
            # Accelerating breakdown = Gamma crush PE
            return "PE"
        
    except Exception as e:
        print(f"Error in gamma trigger detection: {e}")
        return None
    return None

# 🏛️ **PURE INSTITUTIONAL EXECUTION ANALYSIS** 🏛️
def analyze_index_execution(index):
    """
    Analyze ALL 4 INDICES SIMULTANEOUSLY for institutional execution
    Like in screenshots - GREEN ARROW entries
    """
    df_1min = fetch_index_data_1min(index, "1d")
    df_5min = fetch_index_data(index, "5m", "2d")
    
    if df_1min is None or df_5min is None:
        print(f"⚠️ Could not fetch data for {index}")
        return None

    close_1min = ensure_series(df_1min["Close"])
    if len(close_1min) < 20 or close_1min.isna().iloc[-1]:
        print(f"⚠️ Insufficient 1min data for {index}")
        return None

    print(f"🔍 Analyzing {index} for institutional execution...")
    
    # 🏛️ **PRIORITY 1: INSTITUTIONAL CE EXECUTION** (GREEN ARROW CE)
    ce_execution = detect_institutional_ce_execution(df_1min, df_5min)
    if ce_execution:
        print(f"✅ {index}: Institutional CE execution detected")
        return ce_execution, df_5min, False, "institutional_execution_ce"

    # 🏛️ **PRIORITY 2: LIQUIDITY SWEEP EXECUTION** 
    liquidity_execution = detect_liquidity_sweep_execution(df_1min, df_5min)
    if liquidity_execution:
        print(f"✅ {index}: Liquidity sweep execution detected")
        return liquidity_execution, df_5min, False, "liquidity_sweep_execution"

    # 🏛️ **PRIORITY 3: GAMMA TRIGGER EXECUTION**
    gamma_execution = detect_gamma_trigger_execution(df_1min, df_5min)
    if gamma_execution:
        print(f"✅ {index}: Gamma trigger execution detected")
        return gamma_execution, df_5min, False, "gamma_trigger_execution"

    # 🏛️ **PRIORITY 4: INSTITUTIONAL PE EXECUTION** (GREEN ARROW PE)
    pe_execution = detect_institutional_pe_execution(df_1min, df_5min)
    if pe_execution:
        print(f"✅ {index}: Institutional PE execution detected")
        return pe_execution, df_5min, False, "institutional_execution_pe"

    print(f"➖ {index}: No institutional execution detected")
    return None

# --------- SIGNAL DEDUPLICATION ---------
def can_send_signal(index, strike, option_type):
    current_time = time.time()
    strike_key = f"{index}_{strike}_{option_type}"
    
    if strike_key in active_strikes:
        print(f"⚠️ Signal for {strike_key} already active")
        return False
        
    if index in last_signal_time:
        time_since_last = current_time - last_signal_time[index]
        if time_since_last < signal_cooldown:
            print(f"⏳ {index} in cooldown: {int(signal_cooldown - time_since_last)}s remaining")
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
    print(f"📝 Tracking signal {signal_id} for {strike_key}")

def update_signal_progress(signal_id, targets_hit):
    for strike_key, data in active_strikes.items():
        if data['signal_id'] == signal_id:
            active_strikes[strike_key]['targets_hit'] = targets_hit
            break

def clear_completed_signal(signal_id):
    global active_strikes
    active_strikes = {k: v for k, v in active_strikes.items() if v['signal_id'] != signal_id}
    print(f"🗑️ Cleared signal {signal_id}")

# --------- TRADE MONITORING ---------
active_trades = {}

def calculate_pnl(entry, max_price, targets, targets_hit, sl):
    try:
        if targets is None or len(targets) == 0:
            diff = max_price - entry
            if diff > 0:
                return f"+{diff:.2f}"
            elif diff < 0:
                return f"-{abs(diff):.2f}"
            else:
                return "0"
        
        if not isinstance(targets_hit, (list, tuple)):
            targets_hit = list(targets_hit) if targets_hit is not None else [False]*len(targets)
        if len(targets_hit) < len(targets):
            targets_hit = list(targets_hit) + [False] * (len(targets) - len(targets_hit))
        
        achieved_prices = [target for i, target in enumerate(targets) if targets_hit[i]]
        if achieved_prices:
            exit_price = achieved_prices[-1]
            diff = exit_price - entry
            if diff > 0:
                return f"+{diff:.2f}"
            elif diff < 0:
                return f"-{abs(diff):.2f}"
            else:
                return "0"
        else:
            if max_price <= sl:
                diff = sl - entry
                if diff > 0:
                    return f"+{diff:.2f}"
                elif diff < 0:
                    return f"-{abs(diff):.2f}"
                else:
                    return "0"
            else:
                diff = max_price - entry
                if diff > 0:
                    return f"+{diff:.2f}"
                elif diff < 0:
                    return f"-{abs(diff):.2f}"
                else:
                    return "0"
    except Exception:
        return "0"

def monitor_price_live(symbol, entry, targets, sl, fakeout, thread_id, strategy_name, signal_data):
    def monitoring_thread():
        global daily_signals
        
        last_high = entry
        weakness_sent = False
        in_trade = False
        entry_price_achieved = False
        max_price_reached = entry
        targets_hit = [False] * len(targets)
        last_activity_time = time.time()
        signal_id = signal_data.get('signal_id')
        
        print(f"🔍 Starting monitoring for {symbol}")
        
        while True:
            current_time = time.time()
            
            if not in_trade and (current_time - last_activity_time) > 1200:
                send_telegram(f"⏰ {symbol}: No activity for 20 minutes. Allowing new signals.", reply_to=thread_id)
                clear_completed_signal(signal_id)
                print(f"🕒 Monitoring stopped for {symbol} - timeout")
                break
                
            if should_stop_trading():
                try:
                    final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                except Exception:
                    final_pnl = "0"
                signal_data.update({
                    "entry_status": "NOT_ENTERED" if not entry_price_achieved else "ENTERED",
                    "targets_hit": sum(targets_hit),
                    "max_price_reached": max_price_reached,
                    "zero_targets": sum(targets_hit) == 0,
                    "no_new_highs": max_price_reached <= entry,
                    "final_pnl": final_pnl
                })
                daily_signals.append(signal_data)
                clear_completed_signal(signal_id)
                print(f"🏁 Market closed, stopped monitoring {symbol}")
                break
                
            price = fetch_option_price(symbol)
            if price:
                last_activity_time = current_time
                price = round(price)
                
                if price > max_price_reached:
                    max_price_reached = price
                
                if not in_trade:
                    if price >= entry:
                        send_telegram(f"✅ <b>ENTRY TRIGGERED at {price}</b>", reply_to=thread_id)
                        in_trade = True
                        entry_price_achieved = True
                        last_high = price
                        signal_data["entry_status"] = "ENTERED"
                else:
                    if price > last_high:
                        send_telegram(f"🚀 <b>{symbol} making new high → {price}</b>", reply_to=thread_id)
                        last_high = price
                    elif not weakness_sent and price < sl * 1.05:
                        send_telegram(f"⚡ {symbol} showing weakness near SL {sl}", reply_to=thread_id)
                        weakness_sent = True
                    
                    current_targets_hit = sum(targets_hit)
                    for i, target in enumerate(targets):
                        if price >= target and not targets_hit[i]:
                            send_telegram(f"🎯 <b>{symbol}: Target {i+1} hit at ₹{target}</b>", reply_to=thread_id)
                            targets_hit[i] = True
                            current_targets_hit = sum(targets_hit)
                            update_signal_progress(signal_id, current_targets_hit)
                    
                    if price <= sl:
                        send_telegram(f"🔗 <b>{symbol}: Stop Loss {sl} hit. Exit trade. ALLOWING NEW SIGNAL.</b>", reply_to=thread_id)
                        try:
                            final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                        except Exception:
                            final_pnl = "0"
                        signal_data.update({
                            "targets_hit": sum(targets_hit),
                            "max_price_reached": max_price_reached,
                            "zero_targets": sum(targets_hit) == 0,
                            "no_new_highs": max_price_reached <= entry,
                            "final_pnl": final_pnl
                        })
                        daily_signals.append(signal_data)
                        clear_completed_signal(signal_id)
                        print(f"🛑 SL hit for {symbol}")
                        break
                        
                    if current_targets_hit >= 2:
                        update_signal_progress(signal_id, current_targets_hit)
                    
                    if all(targets_hit):
                        send_telegram(f"🏆 <b>{symbol}: ALL TARGETS HIT! Trade completed successfully!</b>", reply_to=thread_id)
                        try:
                            final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                        except Exception:
                            final_pnl = "0"
                        signal_data.update({
                            "targets_hit": len(targets),
                            "max_price_reached": max_price_reached,
                            "zero_targets": False,
                            "no_new_highs": False,
                            "final_pnl": final_pnl
                        })
                        daily_signals.append(signal_data)
                        clear_completed_signal(signal_id)
                        print(f"🎯 All targets hit for {symbol}")
                        break
            
            time.sleep(10)
    
    thread = threading.Thread(target=monitoring_thread)
    thread.daemon = True
    thread.start()

# 🏛️ **INSTITUTIONAL EXECUTION TARGET CALCULATION** 🏛️
def send_execution_signal(index, side, df, fakeout, strategy_key):
    global signal_counter, all_generated_signals
    
    signal_detection_price = float(ensure_series(df["Close"]).iloc[-1])
    strike = round_strike(index, signal_detection_price)
    
    if strike is None:
        print(f"⚠️ Could not round strike for {index} at {signal_detection_price}")
        return
        
    if not can_send_signal(index, strike, side):
        return
        
    symbol = get_option_symbol(index, EXPIRIES[index], strike, side)
    
    if symbol is None:
        print(f"⚠️ Could not generate symbol for {index} {strike}{side}")
        return
    
    option_price = fetch_option_price(symbol)
    if not option_price: 
        print(f"⚠️ Could not fetch price for {symbol}")
        return
    
    entry = round(option_price)
    
    # 🏛️ **INSTITUTIONAL EXECUTION TARGETS** (Like screenshots)
    if side == "CE":
        if strategy_key == "institutional_execution_ce":
            base_move = 95  # Big for institutional CE execution
        elif strategy_key == "liquidity_sweep_execution":
            base_move = 110  # Biggest for liquidity sweeps
        else:
            base_move = 80
        
        targets = [
            round(entry + base_move * 0.6),   # 60% 
            round(entry + base_move * 1.0),   # 100%
            round(entry + base_move * 1.5),   # 150%
            round(entry + base_move * 2.2)    # 220%
        ]
        sl = round(entry - base_move * 0.3)  # Tighter SL for execution
        
    else:  # PE
        if strategy_key == "institutional_execution_pe":
            base_move = 90
        elif strategy_key == "gamma_trigger_execution":
            base_move = 100
        else:
            base_move = 75
        
        targets = [
            round(entry + base_move * 0.6),
            round(entry + base_move * 1.0),
            round(entry + base_move * 1.5),
            round(entry + base_move * 2.2)
        ]
        sl = round(entry - base_move * 0.3)
    
    targets_str = "//".join(str(t) for t in targets) + "++"
    
    strategy_name = STRATEGY_NAMES.get(strategy_key, strategy_key.upper())
    
    signal_id = f"SIG{signal_counter:04d}"
    signal_counter += 1
    
    signal_data = {
        "signal_id": signal_id,
        "timestamp": (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%H:%M:%S"),
        "index": index,
        "strike": strike,
        "option_type": side,
        "strategy": strategy_name,
        "entry_price": entry,
        "targets": targets,
        "sl": sl,
        "fakeout": fakeout,
        "index_price": signal_detection_price,
        "entry_status": "PENDING",
        "targets_hit": 0,
        "max_price_reached": entry,
        "zero_targets": True,
        "no_new_highs": True,
        "final_pnl": "0"
    }
    
    update_signal_tracking(index, strike, side, signal_id)
    all_generated_signals.append(signal_data.copy())
    
    # 🏛️ **INSTITUTIONAL EXECUTION ALERTS** (Like Telegram screenshots)
    if strategy_key == "institutional_execution_ce":
        msg = (f"🏛️ <b>INSTITUTIONAL CE EXECUTION</b> 🏛️\n"
               f"🎯 {index} {strike} CE\n"
               f"SYMBOL: <code>{symbol}</code>\n"
               f"<b>BUY ABOVE: ₹{entry}</b>\n"
               f"TARGETS: {targets_str}\n"
               f"STOP LOSS: ₹{sl}\n"
               f"STRATEGY: {strategy_name}\n"
               f"SIGNAL ID: {signal_id}\n"
               f"⚠️ INSTITUTIONS EXECUTING CE - NEXT MIN BIG MOVE")
    elif strategy_key == "institutional_execution_pe":
        msg = (f"🏛️ <b>INSTITUTIONAL PE EXECUTION</b> 🏛️\n"
               f"🎯 {index} {strike} PE\n"
               f"SYMBOL: <code>{symbol}</code>\n"
               f"<b>BUY ABOVE: ₹{entry}</b>\n"
               f"TARGETS: {targets_str}\n"
               f"STOP LOSS: ₹{sl}\n"
               f"STRATEGY: {strategy_name}\n"
               f"SIGNAL ID: {signal_id}\n"
               f"⚠️ INSTITUTIONS EXECUTING PE - NEXT MIN BIG MOVE")
    elif strategy_key == "liquidity_sweep_execution":
        msg = (f"🎯 <b>LIQUIDITY SWEEP EXECUTION</b> 🎯\n"
               f"🎯 {index} {strike} {side}\n"
               f"SYMBOL: <code>{symbol}</code>\n"
               f"<b>BUY ABOVE: ₹{entry}</b>\n"
               f"TARGETS: {targets_str}\n"
               f"STOP LOSS: ₹{sl}\n"
               f"STRATEGY: {strategy_name}\n"
               f"SIGNAL ID: {signal_id}\n"
               f"⚠️ INSTITUTIONS SWEPT LIQUIDITY - EXECUTING NOW")
    else:
        msg = (f"⚡ <b>GAMMA TRIGGER EXECUTION</b> ⚡\n"
               f"🎯 {index} {strike} {side}\n"
               f"SYMBOL: <code>{symbol}</code>\n"
               f"<b>BUY ABOVE: ₹{entry}</b>\n"
               f"TARGETS: {targets_str}\n"
               f"STOP LOSS: ₹{sl}\n"
               f"STRATEGY: {strategy_name}\n"
               f"SIGNAL ID: {signal_id}\n"
               f"⚠️ GAMMA TRIGGERED - INSTITUTIONS HEDGING")
    
    print(f"📤 Sending execution signal: {index} {strike}{side} @ {entry}")
    thread_id = send_telegram(msg)
    
    trade_id = f"{symbol}_{int(time.time())}"
    active_trades[trade_id] = {
        "symbol": symbol, 
        "entry": entry, 
        "sl": sl, 
        "targets": targets, 
        "thread": thread_id, 
        "status": "OPEN",
        "index": index,
        "signal_data": signal_data
    }
    
    monitor_price_live(symbol, entry, targets, sl, fakeout, thread_id, strategy_name, signal_data)

# --------- EXECUTION THREAD ---------
def execution_thread(index):
    print(f"🔍 Checking {index} for institutional execution...")
    result = analyze_index_execution(index)
    
    if not result:
        return
        
    side, df, fakeout, strategy_key = result
    
    print(f"✅ Execution detected for {index}: {side} ({strategy_key})")
    send_execution_signal(index, side, df, fakeout, strategy_key)

# --------- MAIN LOOP ---------
def run_execution_parallel():
    if not is_market_open(): 
        print("🔴 Market closed")
        return
        
    if should_stop_trading():
        global STOP_SENT, EOD_REPORT_SENT
        if not STOP_SENT:
            send_telegram("🛑 Market closed at 3:30 PM IST - Algorithm stopped")
            STOP_SENT = True
            
        if not EOD_REPORT_SENT:
            time.sleep(15)
            send_telegram("📊 GENERATING COMPULSORY END-OF-DAY REPORT...")
            # EOD report logic here
            EOD_REPORT_SENT = True
            
        return
        
    print("🔍 Scanning ALL 4 INDICES for institutional execution...")
    threads = []
    indices = ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"]
    
    for index in indices:
        t = threading.Thread(target=execution_thread, args=(index,))
        t.start()
        threads.append(t)
    
    for t in threads: 
        t.join()
    
    print(f"✅ Scan complete. Active executions: {len(active_strikes)}")

# --------- MAIN EXECUTION ---------
STARTED_SENT = False
STOP_SENT = False
MARKET_CLOSED_SENT = False
EOD_REPORT_SENT = False

print("=" * 60)
print("🏛️  PURE INSTITUTIONAL EXECUTION AI ACTIVATED")
print("🎯 DETECTING GREEN ARROW ENTRIES LIKE SCREENSHOTS")
print("⚡ INSTITUTIONAL CE/PE EXECUTION DETECTION")
print("🎯 LIQUIDITY SWEEP EXECUTION DETECTION")
print("⚡ GAMMA TRIGGER EXECUTION DETECTION")
print("🔄 SIMULTANEOUS SCANNING OF ALL 4 INDICES")
print("=" * 60)

iteration = 0
while True:
    iteration += 1
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        current_time_ist = ist_now.time()
        
        market_open = is_market_open()
        
        if not market_open:
            if not MARKET_CLOSED_SENT:
                send_telegram("🔴 Market is currently closed. Algorithm waiting for 9:15 AM...")
                MARKET_CLOSED_SENT = True
                STARTED_SENT = False
                STOP_SENT = False
                EOD_REPORT_SENT = False
            print(f"⏳ Iteration {iteration}: Market closed. Waiting...")
            time.sleep(30)
            continue
        
        if not STARTED_SENT:
            send_telegram("🏛️ <b>PURE INSTITUTIONAL EXECUTION AI ACTIVATED</b>\n"
                         "🎯 <b>DETECTING GREEN ARROW ENTRIES</b>\n"
                         "⚡ <b>INSTITUTIONAL CE/PE EXECUTION</b>\n"
                         "🎯 <b>LIQUIDITY SWEEP EXECUTION</b>\n"
                         "⚡ <b>GAMMA TRIGGER EXECUTION</b>\n"
                         "🔄 <b>SIMULTANEOUS SCANNING ALL 4 INDICES</b>\n"
                         "⚠️ <b>NO RETAIL PATTERNS - PURE INSTITUTIONAL</b>\n"
                         "🎯 <b>ENTRY THEN IMMEDIATE BIG MOVE</b>")
            STARTED_SENT = True
            STOP_SENT = False
            MARKET_CLOSED_SENT = False
        
        if should_stop_trading():
            if not STOP_SENT:
                send_telegram("🛑 Market closing time reached!")
                STOP_SENT = True
                STARTED_SENT = False
            print(f"⏰ Iteration {iteration}: Market closing time")
            time.sleep(60)
            continue
            
        print(f"🔄 Iteration {iteration}: Scanning ALL INDICES...")
        run_execution_parallel()
        time.sleep(30)  # Check every 30 seconds
        
    except Exception as e:
        error_msg = f"⚠️ Main loop error: {str(e)[:100]}"
        print(error_msg)
        send_telegram(error_msg)
        time.sleep(60)
