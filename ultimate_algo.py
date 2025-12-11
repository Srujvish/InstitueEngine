# INSTITUTIONAL INTRADAY LEVELS WITH ANGEL ONE SIGNALS
# REAL-TIME MARKET MAKER FLOW

import os
import time
import requests
import pandas as pd
import yfinance as yf
import warnings
import numpy as np
from datetime import datetime, timedelta, time as dtime
import pytz
import pyotp
from SmartApi.smartConnect import SmartConnect
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

# ==================== ANGEL ONE SETUP ====================
API_KEY = os.getenv("API_KEY")
CLIENT_CODE = os.getenv("CLIENT_CODE")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")

# Initialize Angel One client
try:
    client = SmartConnect(api_key=API_KEY)
    TOTP = pyotp.TOTP(TOTP_SECRET).now()
    session = client.generateSession(CLIENT_CODE, PASSWORD, TOTP)
    feedToken = client.getfeedToken()
    print("✅ Angel One login successful")
except Exception as e:
    print(f"❌ Angel One login failed: {e}")
    client = None

# ==================== TELEGRAM SETUP ====================
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram(msg: str, parse_mode: str = "HTML") -> bool:
    """Send message to Telegram"""
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": msg,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

# ==================== TIME FUNCTIONS ====================
def get_ist_time():
    """Get current IST time"""
    utc_now = datetime.utcnow()
    ist = pytz.timezone('Asia/Kolkata')
    return utc_now.replace(tzinfo=pytz.utc).astimezone(ist)

def is_market_open():
    """Check if market is open"""
    ist = get_ist_time()
    current_time = ist.time()
    current_day = ist.weekday()
    return (dtime(9, 15) <= current_time <= dtime(15, 30) and current_day <= 4)

def is_pre_market():
    """Check if pre-market hours (9:00-9:15)"""
    ist = get_ist_time()
    current_time = ist.time()
    return dtime(9, 0) <= current_time < dtime(9, 15)

# ==================== INSTITUTIONAL LEVELS CALCULATION ====================
class InstitutionalLevels:
    """Calculate institutional support/resistance levels"""
    
    def __init__(self):
        self.cached_levels = {}
        
    def fetch_historical_data(self, symbol: str, days: int = 3):
        """Fetch historical data from Yahoo Finance"""
        ticker_map = {
            "NIFTY": "^NSEI",
            "BANKNIFTY": "^NSEBANK",
            "FINNIFTY": "^NSETFIN"
        }
        
        try:
            ticker = ticker_map.get(symbol, symbol)
            df = yf.download(ticker, period=f"{days}d", interval="15m", progress=False)
            return df if not df.empty else None
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def calculate_value_area(self, df: pd.DataFrame):
        """Calculate Value Area High/Low and POC"""
        if df is None or len(df) < 10:
            return None, None, None
        
        try:
            # Calculate volume profile
            price_bins = np.linspace(df['Low'].min(), df['High'].max(), 50)
            volume_profile = []
            
            for i in range(len(price_bins)-1):
                mask = (df['Close'] >= price_bins[i]) & (df['Close'] < price_bins[i+1])
                volume_in_bin = df.loc[mask, 'Volume'].sum()
                volume_profile.append({
                    'price': (price_bins[i] + price_bins[i+1]) / 2,
                    'volume': volume_in_bin
                })
            
            # Sort by volume
            volume_profile = sorted(volume_profile, key=lambda x: x['volume'], reverse=True)
            
            # Find POC (Point of Control)
            poc = volume_profile[0]['price'] if volume_profile else None
            
            # Calculate Value Area (70% of volume)
            total_volume = sum(item['volume'] for item in volume_profile)
            target_volume = total_volume * 0.70
            
            cumulative_volume = 0
            value_area_prices = []
            
            for item in volume_profile:
                cumulative_volume += item['volume']
                value_area_prices.append(item['price'])
                if cumulative_volume >= target_volume:
                    break
            
            value_area_high = max(value_area_prices) if value_area_prices else None
            value_area_low = min(value_area_prices) if value_area_prices else None
            
            return value_area_high, value_area_low, poc
            
        except Exception as e:
            print(f"Error calculating value area: {e}")
            return None, None, None
    
    def get_yesterday_levels(self, symbol: str):
        """Get yesterday's key levels"""
        try:
            df = self.fetch_historical_data(symbol, 3)
            if df is None:
                return None
            
            # Get yesterday's date
            ist = get_ist_time()
            yesterday_date = (ist.date() - timedelta(days=1))
            
            # Filter yesterday's data
            df['Date'] = df.index.date
            yesterday_df = df[df['Date'] == yesterday_date]
            
            if yesterday_df.empty:
                return None
            
            # Basic levels
            yest_high = float(yesterday_df['High'].max())
            yest_low = float(yesterday_df['Low'].min())
            yest_close = float(yesterday_df['Close'].iloc[-1])
            
            # Value Area levels
            vah, val, poc = self.calculate_value_area(yesterday_df)
            
            return {
                'high': yest_high,
                'low': yest_low,
                'close': yest_close,
                'value_area_high': vah,
                'value_area_low': val,
                'poc': poc
            }
            
        except Exception as e:
            print(f"Error getting yesterday levels: {e}")
            return None
    
    def get_current_price(self, symbol: str):
        """Get current price from Angel One"""
        if client is None:
            return None
            
        try:
            # Map symbols
            symbol_map = {
                "NIFTY": "NIFTY",
                "BANKNIFTY": "BANKNIFTY"
            }
            
            # For indices, we need to use appropriate token
            # This is a simplified version - you need to implement proper token mapping
            token_map = self.load_token_map()
            symbol_key = f"{symbol}-INDEX"
            
            if symbol_key in token_map:
                ltp_data = client.ltpData("NSE", symbol, token_map[symbol_key])
                return float(ltp_data['data']['ltp'])
            else:
                # Fallback to Yahoo Finance
                ticker = "^NSEI" if symbol == "NIFTY" else "^NSEBANK"
                df = yf.download(ticker, period="1d", interval="1m", progress=False)
                return float(df['Close'].iloc[-1]) if not df.empty else None
                
        except Exception as e:
            print(f"Error getting current price: {e}")
            return None
    
    def load_token_map(self):
        """Load token map from Angel One"""
        try:
            url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            token_map = {}
            for item in data:
                symbol = item.get('symbol', '').upper()
                token = item.get('token', '')
                
                if symbol and token:
                    token_map[symbol] = token
            
            return token_map
        except Exception as e:
            print(f"Error loading token map: {e}")
            return {}
    
    def calculate_today_levels(self, symbol: str):
        """Calculate today's institutional levels"""
        try:
            # Get yesterday's levels
            yest_levels = self.get_yesterday_levels(symbol)
            if yest_levels is None:
                return None
            
            # Get current price
            current_price = self.get_current_price(symbol)
            if current_price is None:
                # Estimate opening price
                current_price = yest_levels['close']
            
            # Calculate overnight gap
            gap = current_price - yest_levels['close']
            gap_percent = (gap / yest_levels['close']) * 100
            
            # Calculate today's levels based on gap
            if gap_percent < -0.5:  # Gap down
                # Yesterday's support becomes resistance
                resistance1 = yest_levels['value_area_low'] or yest_levels['low']
                resistance2 = yest_levels['poc'] or (yest_levels['high'] + yest_levels['low']) / 2
                
                # New support based on gap
                support1 = current_price * 0.995  # 0.5% below
                support2 = current_price * 0.99   # 1% below
                
            elif gap_percent > 0.5:  # Gap up
                # Yesterday's resistance becomes support
                support1 = yest_levels['value_area_high'] or yest_levels['high']
                support2 = yest_levels['poc'] or (yest_levels['high'] + yest_levels['low']) / 2
                
                # New resistance based on gap
                resistance1 = current_price * 1.005  # 0.5% above
                resistance2 = current_price * 1.01   # 1% above
                
            else:  # Small gap
                # Use Value Area levels
                if yest_levels['value_area_high'] and yest_levels['value_area_low']:
                    resistance1 = yest_levels['value_area_high']
                    support1 = yest_levels['value_area_low']
                else:
                    resistance1 = yest_levels['high']
                    support1 = yest_levels['low']
                
                resistance2 = resistance1 * 1.005
                support2 = support1 * 0.995
            
            # Ensure levels are realistic (not too far)
            max_distance = current_price * 0.015  # Max 1.5% away
            min_distance = current_price * 0.003  # Min 0.3% away
            
            # Adjust levels to be within range
            for level_name in ['resistance1', 'resistance2', 'support1', 'support2']:
                level = locals()[level_name]
                distance = abs(level - current_price)
                
                if distance > max_distance:
                    # Bring level closer
                    direction = 1 if level > current_price else -1
                    locals()[level_name] = current_price + (direction * max_distance)
                elif distance < min_distance:
                    # Push level further
                    direction = 1 if level > current_price else -1
                    locals()[level_name] = current_price + (direction * min_distance)
            
            # Round to appropriate intervals
            round_to = 50 if symbol == "NIFTY" else 100 if symbol == "BANKNIFTY" else 10
            
            levels = {
                'current_price': round(current_price, 2),
                'gap': round(gap, 2),
                'gap_percent': round(gap_percent, 2),
                'yesterday_high': round(yest_levels['high'], 2),
                'yesterday_low': round(yest_levels['low'], 2),
                'yesterday_close': round(yest_levels['close'], 2),
                'yesterday_poc': round(yest_levels['poc'], 2) if yest_levels['poc'] else None,
                'support1': round(support1 / round_to) * round_to,
                'support2': round(support2 / round_to) * round_to,
                'resistance1': round(resistance1 / round_to) * round_to,
                'resistance2': round(resistance2 / round_to) * round_to
            }
            
            return levels
            
        except Exception as e:
            print(f"Error calculating today levels: {e}")
            return None

# ==================== SIGNAL GENERATION ====================
class InstitutionalSignals:
    """Generate signals based on institutional levels"""
    
    def __init__(self):
        self.level_detector = InstitutionalLevels()
        self.active_signals = {}
        
    def check_for_signal(self, symbol: str):
        """Check if price is at institutional level and generate signal"""
        try:
            # Get current price
            current_price = self.level_detector.get_current_price(symbol)
            if current_price is None:
                return None
            
            # Get today's levels
            levels = self.level_detector.calculate_today_levels(symbol)
            if levels is None:
                return None
            
            # Check if price is near any institutional level
            signal = None
            level_type = None
            
            # Check support levels
            for i in [1, 2]:
                support_key = f'support{i}'
                if support_key in levels and levels[support_key]:
                    distance_pct = abs(current_price - levels[support_key]) / current_price
                    if distance_pct <= 0.0015:  # Within 0.15%
                        signal = "BUY"
                        level_type = f"Support {i}"
                        level_price = levels[support_key]
                        break
            
            # Check resistance levels
            if not signal:
                for i in [1, 2]:
                    resistance_key = f'resistance{i}'
                    if resistance_key in levels and levels[resistance_key]:
                        distance_pct = abs(current_price - levels[resistance_key]) / current_price
                        if distance_pct <= 0.0015:  # Within 0.15%
                            signal = "SELL"
                            level_type = f"Resistance {i}"
                            level_price = levels[resistance_key]
                            break
            
            # Check POC level
            if not signal and levels.get('yesterday_poc'):
                distance_pct = abs(current_price - levels['yesterday_poc']) / current_price
                if distance_pct <= 0.001:  # Within 0.1%
                    # Determine direction based on gap
                    if levels['gap'] > 0:
                        signal = "BUY"  # Price above POC in gap up
                    else:
                        signal = "SELL"  # Price below POC in gap down
                    level_type = "POC"
                    level_price = levels['yesterday_poc']
            
            if signal:
                # Get option chain data for confirmation
                option_data = self.get_option_chain_data(symbol)
                
                signal_info = {
                    'symbol': symbol,
                    'signal': signal,
                    'level_type': level_type,
                    'level_price': level_price,
                    'current_price': current_price,
                    'distance_pct': distance_pct,
                    'timestamp': get_ist_time().strftime("%H:%M:%S"),
                    'option_data': option_data
                }
                
                return signal_info
            
            return None
            
        except Exception as e:
            print(f"Error checking for signal: {e}")
            return None
    
    def get_option_chain_data(self, symbol: str):
        """Get option chain data from Angel One"""
        if client is None:
            return None
            
        try:
            # This is simplified - you need to implement proper option chain fetching
            # based on your Angel One setup
            
            # For now, return dummy data
            return {
                'max_pain': None,
                'ce_oi_change': 0,
                'pe_oi_change': 0
            }
            
        except Exception as e:
            print(f"Error getting option data: {e}")
            return None
    
    def generate_trade_signal(self, signal_info: Dict):
        """Generate complete trade signal with entry, targets, SL"""
        if not signal_info:
            return None
        
        symbol = signal_info['symbol']
        signal = signal_info['signal']
        current_price = signal_info['current_price']
        
        # Calculate trade parameters based on symbol
        if symbol == "NIFTY":
            point_value = 50
            stop_distance = 40  # 40 points
            target1_distance = 60  # 60 points
            target2_distance = 100  # 100 points
        elif symbol == "BANKNIFTY":
            point_value = 100
            stop_distance = 80  # 80 points
            target1_distance = 120  # 120 points
            target2_distance = 200  # 200 points
        else:
            point_value = 10
            stop_distance = 20  # 20 points
            target1_distance = 30  # 30 points
            target2_distance = 50  # 50 points
        
        if signal == "BUY":
            entry = current_price + (point_value * 0.5)  # Slightly above level
            stop_loss = entry - (stop_distance * point_value)
            target1 = entry + (target1_distance * point_value)
            target2 = entry + (target2_distance * point_value)
        else:  # SELL
            entry = current_price - (point_value * 0.5)  # Slightly below level
            stop_loss = entry + (stop_distance * point_value)
            target1 = entry - (target1_distance * point_value)
            target2 = entry - (target2_distance * point_value)
        
        # Find nearest option strike
        strike = self.get_nearest_strike(symbol, entry)
        
        trade_signal = {
            'symbol': symbol,
            'action': signal,
            'strike': strike,
            'entry': round(entry, 2),
            'stop_loss': round(stop_loss, 2),
            'target1': round(target1, 2),
            'target2': round(target2, 2),
            'level_type': signal_info['level_type'],
            'level_price': signal_info['level_price'],
            'current_price': current_price,
            'timestamp': signal_info['timestamp'],
            'risk_reward': round((target1 - entry) / (entry - stop_loss), 2) if signal == "BUY" else round((entry - target1) / (stop_loss - entry), 2)
        }
        
        return trade_signal
    
    def get_nearest_strike(self, symbol: str, price: float):
        """Get nearest option strike"""
        if symbol == "NIFTY":
            strike_interval = 50
        elif symbol == "BANKNIFTY":
            strike_interval = 100
        else:
            strike_interval = 50
        
        nearest = round(price / strike_interval) * strike_interval
        return int(nearest)

# ==================== MAIN EXECUTION ====================
def send_pre_market_report():
    """Send pre-market institutional levels report"""
    ist = get_ist_time()
    
    report = []
    report.append(f"📊 <b>PRE-MARKET INSTITUTIONAL LEVELS</b>")
    report.append(f"📅 {ist.strftime('%d %b %Y, %A')}")
    report.append(f"⏰ {ist.strftime('%H:%M IST')}")
    report.append("")
    report.append("══════════════════════════════")
    report.append("")
    
    level_detector = InstitutionalLevels()
    
    for symbol in ["NIFTY", "BANKNIFTY"]:
        levels = level_detector.calculate_today_levels(symbol)
        
        if levels:
            report.append(f"<b>📈 {symbol}</b>")
            report.append(f"• Current/Expected Open: <b>{levels['current_price']}</b>")
            report.append(f"• Gap: <code>{levels['gap']:+.0f} pts ({levels['gap_percent']:+.2f}%)</code>")
            report.append("")
            
            report.append(f"<b>🛡️ INSTITUTIONAL SUPPORT (Buy Zones):</b>")
            report.append(f"  S1: <code>{levels['support1']}</code>")
            report.append(f"  S2: <code>{levels['support2']}</code>")
            report.append("")
            
            report.append(f"<b>🎯 INSTITUTIONAL RESISTANCE (Sell Zones):</b>")
            report.append(f"  R1: <code>{levels['resistance1']}</code>")
            report.append(f"  R2: <code>{levels['resistance2']}</code>")
            report.append("")
            
            report.append(f"<b>📊 YESTERDAY'S REFERENCE:</b>")
            report.append(f"  High: <code>{levels['yesterday_high']}</code>")
            report.append(f"  Low: <code>{levels['yesterday_low']}</code>")
            report.append(f"  Close: <code>{levels['yesterday_close']}</code>")
            if levels['yesterday_poc']:
                report.append(f"  POC: <code>{levels['yesterday_poc']}</code>")
            
            report.append("")
            report.append("══════════════════════════════")
            report.append("")
    
    report.append("<b>🎯 TRADING PLAN:</b>")
    report.append("• Watch for price action at institutional levels")
    report.append("• Enter only with volume confirmation")
    report.append("• Use tight stops (40 pts for Nifty, 80 pts for BankNifty)")
    report.append("• Book partial profits at first target")
    report.append("")
    report.append("<b>⚠️ RISK MANAGEMENT:</b>")
    report.append("• Max 1% risk per trade")
    report.append("• No revenge trading")
    report.append("• Respect stop losses")
    
    full_report = "\n".join(report)
    send_telegram(full_report)
    
    # Save to file
    os.makedirs("reports", exist_ok=True)
    filename = f"reports/institutional_{ist.strftime('%Y%m%d_%H%M')}.txt"
    with open(filename, "w") as f:
        f.write(full_report)
    
    print(f"✅ Pre-market report sent: {filename}")

def monitor_market_signals():
    """Monitor market and generate signals"""
    signal_generator = InstitutionalSignals()
    last_signal_time = {}
    
    print("🚀 Starting institutional signal monitoring...")
    
    while True:
        try:
            ist = get_ist_time()
            current_time = ist.time()
            
            # Stop monitoring after market close
            if current_time > dtime(15, 30):
                print("Market closed. Stopping monitoring.")
                break
            
            # Skip pre-market
            if current_time < dtime(9, 15):
                time.sleep(60)
                continue
            
            for symbol in ["NIFTY", "BANKNIFTY"]:
                # Check cooldown period (5 minutes between signals for same symbol)
                if symbol in last_signal_time:
                    time_since_last = (ist - last_signal_time[symbol]).total_seconds()
                    if time_since_last < 300:  # 5 minutes cooldown
                        continue
                
                # Check for signal
                signal_info = signal_generator.check_for_signal(symbol)
                
                if signal_info:
                    trade_signal = signal_generator.generate_trade_signal(signal_info)
                    
                    if trade_signal:
                        # Format signal message
                        msg = []
                        msg.append(f"🚨 <b>INSTITUTIONAL SIGNAL - {symbol}</b>")
                        msg.append(f"⏰ {ist.strftime('%H:%M:%S IST')}")
                        msg.append("")
                        msg.append(f"<b>Action:</b> {trade_signal['action']}")
                        msg.append(f"<b>Level:</b> {trade_signal['level_type']} at {trade_signal['level_price']}")
                        msg.append(f"<b>Current Price:</b> {trade_signal['current_price']}")
                        msg.append("")
                        msg.append(f"<b>Trade Setup:</b>")
                        msg.append(f"• Strike: {trade_signal['strike']}")
                        msg.append(f"• Entry: {trade_signal['entry']}")
                        msg.append(f"• Stop Loss: {trade_signal['stop_loss']}")
                        msg.append(f"• Target 1: {trade_signal['target1']}")
                        msg.append(f"• Target 2: {trade_signal['target2']}")
                        msg.append(f"• Risk/Reward: 1:{trade_signal['risk_reward']}")
                        msg.append("")
                        msg.append("<b>Confirmation Required:</b>")
                        msg.append("• Wait for price to hold level")
                        msg.append("• Check volume spike")
                        msg.append("• Look for option OI change")
                        
                        full_msg = "\n".join(msg)
                        send_telegram(full_msg)
                        
                        print(f"Signal sent for {symbol}: {trade_signal['action']}")
                        
                        # Update last signal time
                        last_signal_time[symbol] = ist
            
            # Wait before next check
            time.sleep(30)
            
        except Exception as e:
            print(f"Error in monitoring: {e}")
            time.sleep(60)

def main():
    """Main execution function"""
    print("=" * 50)
    print("INSTITUTIONAL INTRADAY LEVELS & SIGNALS")
    print(f"Start Time: {get_ist_time().strftime('%H:%M:%S IST')}")
    print("=" * 50)
    
    try:
        # Step 1: Send pre-market report at 9:00 AM
        ist = get_ist_time()
        
        if is_pre_market():
            print("Sending pre-market report...")
            send_pre_market_report()
            
            # Wait for market to open
            print("Waiting for market to open at 9:15...")
            while is_pre_market():
                time.sleep(30)
        
        # Step 2: Start signal monitoring
        if is_market_open():
            print("Market is open. Starting signal monitoring...")
            monitor_market_signals()
        else:
            print("Market is closed. Exiting.")
            
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    except Exception as e:
        print(f"Main error: {e}")
        send_telegram(f"❌ System error: {str(e)[:100]}")

# ==================== START ====================
if __name__ == "__main__":
    # Send startup message
    startup_msg = f"""
🚀 <b>Institutional Intraday System Started</b>
⏰ {get_ist_time().strftime('%H:%M:%S IST')}
📅 {get_ist_time().strftime('%d %b %Y')}
✅ Angel One: {'Connected' if client else 'Not Connected'}
✅ Telegram: {'Ready' if BOT_TOKEN and CHAT_ID else 'Not Configured'}
"""
    send_telegram(startup_msg)
    
    # Run main program
    main()
