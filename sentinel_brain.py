import os
import json
import datetime
import hashlib
from datetime import timedelta
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import xgboost as xgb
import nltk
from transformers import pipeline

# ======================================================================
# 1. INSTITUTIONAL CONFIGURATION & TIME CORES (V14)
# ======================================================================

TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE" 
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"     

# سبد جامع دارایی‌های جهانی (فارکس، طلا، کریپتو)
# سبد جامع دارایی‌های جهانی (28 جفت‌ارز + طلا + کریپتو)
SYMBOLS = [
    # Majors (ماژورها)
    "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X", "USDCHF=X", "USDJPY=X",
    # Yen Crosses (کراس‌های ین)
    "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "NZDJPY=X", "CADJPY=X", "CHFJPY=X",
    # Euro Crosses (کراس‌های یورو)
    "EURGBP=X", "EURAUD=X", "EURNZD=X", "EURCAD=X", "EURCHF=X",
    # Pound Crosses (کراس‌های پوند)
    "GBPAUD=X", "GBPNZD=X", "GBPCAD=X", "GBPCHF=X",
    # Other Crosses (سایر کراس‌ها)
    "AUDCAD=X", "AUDCHF=X", "AUDNZD=X", "NZDCAD=X", "NZDCHF=X", "CADCHF=X",
    # Commodities & Crypto (کالا و رمزارز)
    "GC=F", "ETH-USD", "LTC-USD", "DOGE-USD"
]


# ریسک پیش‌فرض پایه
DEFAULT_RISK_PERCENT = 1.0 

# ساعات طلایی نقدینگی فارکس (کریپتو 24 ساعته است که در موتور لحاظ می‌شود)
TRADING_HOURS = {
    "london_start": 7, 
    "new_york_end": 21
}

def is_market_liquid_now(symbol):
    """
    بررسی نقدینگی بازار. برای رمزارزها همیشه True است.
    """
    if "-USD" in symbol:
        return True # بازار کریپتو همیشه باز و دارای نقدینگی است
        
    current_hour = datetime.datetime.utcnow().hour
    current_day = datetime.datetime.utcnow().weekday()
    
    # تعطیلات آخر هفته برای فارکس و طلا
    if current_day >= 5: 
        return False
        
    if TRADING_HOURS["london_start"] <= current_hour <= TRADING_HOURS["new_york_end"]:
        return True
        
    return False

def generate_signal_id(symbol, direction, timestamp_str):
    """
    تولید یک هش یکتا برای سیگنال جهت جلوگیری از باز شدن پوزیشن‌های تکراری 
    در اکسپرت MT5. (Anti-Duplicate System)
    """
    raw_string = f"{symbol}_{direction}_{timestamp_str}"
    return hashlib.md5(raw_string.encode()).hexdigest()[:10]

def send_telegram_alert(message):
    """ ارسال نوتیفیکیشن تلگرام """
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE": 
        return 
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID, 
        "text": message, 
        "parse_mode": "HTML"
    }
    
    try: 
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"[ERROR] Telegram alert failed: {e}")

# ======================================================================
# 2. ADVANCED RISK MANAGEMENT (Kelly Criterion & Correlation)
# ======================================================================

def calculate_kelly_criterion(win_probability, tp_dist, sl_dist):
    """ محاسبه حجم بهینه با فرمول Half-Kelly Criterion """
    if sl_dist <= 0 or win_probability < 0.50: 
        return 0.0
        
    b = tp_dist / sl_dist
    p = win_probability
    q = 1.0 - p
    
    kelly_fraction = p - (q / b)
    half_kelly_percent = (kelly_fraction / 2.0) * 100
    
    final_risk = max(min(half_kelly_percent, 3.0), 0.5)
    return round(final_risk, 2)

def calculate_portfolio_correlation(symbols_list, lookback_days=10):
    """ محاسبه ماتریس همبستگی برای جلوگیری از ورود همزمان به دارایی‌های مشابه """
    print("[INFO] Computing Live Pearson Correlation Matrix...")
    close_prices = {}
    
    for sym in symbols_list:
        try:
            df = yf.download(sym, period=f"{lookback_days}d", interval="1h", progress=False)
            if not df.empty:
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                close_prices[sym] = df['Close']
        except Exception: 
            pass
            
    if len(close_prices) < 2: 
        return pd.DataFrame()
        
    correlation_matrix = pd.DataFrame(close_prices).corr(method='pearson')
    print("[SUCCESS] Correlation Matrix Calculated.")
    return correlation_matrix

# ======================================================================
# 3. QUANTITATIVE INDICATORS, FVG & VSA
# ======================================================================

def calculate_vwap(df):
    """ محاسبه میانگین قیمت وزنی حجم (VWAP) """
    q = df['Volume'].replace(0, 1) 
    p = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (p * q).cumsum() / q.cumsum()
    return vwap

def calculate_poc(df, lookback=100):
    """ محاسبه نقطه کنترل حجم (POC) به عنوان آهنربای قیمت """
    if len(df) < lookback: 
        return df['Close'].iloc[-1]
    recent = df.iloc[-lookback:].copy()
    if recent['Volume'].sum() == 0: 
        return recent['Close'].mean()
    recent['Price_Bin'] = recent['Close'].round(4)
    poc_price = recent.groupby('Price_Bin')['Volume'].sum().idxmax()
    return poc_price

def calculate_hurst(price_series, max_lag=20):
    """ محاسبه Hurst Exponent برای تشخیص رژیم بازار (روند یا رنج) """
    if len(price_series) < max_lag: 
        return 0.5
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def calculate_rsi(data, periods=14):
    """ محاسبه اندیکاتور RSI """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_width(df, window=20):
    """ محاسبه فشردگی باندهای بولینگر """
    std = df['Close'].rolling(window=window).std()
    ma = df['Close'].rolling(window=window).mean()
    upper = ma + (std * 2)
    lower = ma - (std * 2)
    bb_width = (upper - lower) / ma
    return bb_width
def detect_volatility_squeeze(df, window=20):
    """
    تشخیص فشردگی بازار (Volatility Squeeze).
    وقتی باند بولینگر به باریک‌ترین حالت خود در 20 کندل اخیر می‌رسد،
    نشان‌دهنده تجمیع سفارشات بانک‌ها و آمادگی برای یک شکست (Breakout) بزرگ است.
    """
    if len(df) < window * 2:
        return False
        
    bb_width = calculate_bollinger_width(df, window)
    current_width = bb_width.iloc[-1]
    lowest_width_recent = bb_width.iloc[-window-5:-1].min()
    
    # اگر فشردگی فعلی نزدیک به کمترین میزان فشردگی اخیر باشد
    if current_width <= (lowest_width_recent * 1.05):
        return True
    return False
def detect_vsa_anomaly(df):
    """ تحلیل حجم و اسپرد (VSA) برای کشف ورود نهنگ‌ها """
    if len(df) < 20: 
        return False
    vol_sma = df['Volume'].rolling(20).mean().iloc[-2]
    current_vol = df['Volume'].iloc[-2]
    if current_vol > (vol_sma * 1.5):
        return True
    return False

def check_liquidity_sweep(df, lookback=20):
    """ بررسی شکار نقدینگی (Stop Hunt) """
    if len(df) < lookback + 2: 
        return "none"
    recent_high = df['High'].iloc[-lookback-2:-2].max()
    recent_low = df['Low'].iloc[-lookback-2:-2].min()
    current = df.iloc[-2] 
    if current['High'] > recent_high and current['Close'] < recent_high: 
        return "bearish_sweep"
    elif current['Low'] < recent_low and current['Close'] > recent_low: 
        return "bullish_sweep"
    return "none"

def detect_fair_value_gaps(df, lookback=15):
    """ کشف گپ‌های ارزش منصفانه (FVG) """
    fvgs = {"bullish": [], "bearish": []}
    if len(df) < lookback: 
        return fvgs
    recent = df.iloc[-lookback:-1]
    for i in range(1, len(recent)-1):
        if recent['Low'].iloc[i+1] > recent['High'].iloc[i-1]:
            if recent['Close'].iloc[i] > recent['Open'].iloc[i]:
                fvgs["bullish"].append((recent['High'].iloc[i-1], recent['Low'].iloc[i+1]))
        if recent['High'].iloc[i+1] < recent['Low'].iloc[i-1]:
            if recent['Close'].iloc[i] < recent['Open'].iloc[i]:
                fvgs["bearish"].append((recent['Low'].iloc[i-1], recent['High'].iloc[i+1]))
    return fvgs



# ======================================================================
# 4. GLOBAL CRISIS NLP ENGINE (FinBERT Deep Learning V14)
# ======================================================================

# بارگذاری مدل هوش مصنوعی در حافظه کش
try:
    print("[INFO] Loading FinBERT Financial NLP Model...")
    # از pipeline هاگینگ‌فیس برای تحلیل احساسات اقتصادی استفاده می‌کنیم
    finbert_nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except Exception as e:
    print(f"[ERROR] Failed to load FinBERT: {e}")
    finbert_nlp = None

def analyze_crisis_news():
    """
    تحلیل عمیق اخبار با استفاده از شبکه عصبی FinBERT.
    پشتیبانی کامل از فارکس، فلزات گرانبها و کریپتوکارنسی.
    """
    print("[INFO] Initializing V14 Deep NLP Matrix...")
    
    # منابع خبری جامع (فارکس + کریپتو + کلان اقتصادی)
    urls = [
        "https://www.fxstreet.com/rss/news",               # Forex & Gold
        "https://cointelegraph.com/rss",                   # Crypto Heavy
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664" # Global Economy
    ]
    
    # دیکشنری پیشرفته کلمات کلیدی برای تمامی دارایی‌ها
    asset_keywords = {
        "USD": ["FED", "FOMC", "CPI", "NFP", "POWELL", "US ", "TREASURY", "DOLLAR"],
        "EUR": ["ECB", "LAGARDE", "EUROZONE", "GERMANY", "FRANCE"],
        "GBP": ["BOE", "BAILEY", "UK ", "BRITAIN", "BREXIT"],
        "AUD": ["RBA", "AUSTRALIA", "AUSSIE", "SYDNEY"],
        "CAD": ["BOC", "CANADA", "LOONIE", "OTTAWA", "OIL"],
        "JPY": ["BOJ", "YEN", "JAPAN", "TOKYO", "UEDA"],
        "CHF": ["SNB", "SWISS", "FRANC", "ZURICH"],
        "XAU": ["GOLD", "OUNCE", "BULLION", "SAFE HAVEN", "XAU"],
        "ETH": ["ETHEREUM", "ETH", "VITALIK", "SMART CONTRACT", "GAS FEE"],
        "LTC": ["LITECOIN", "LTC", "CHARLIE LEE"],
        "DOGE": ["DOGECOIN", "DOGE", "ELON", "MUSK", "MEMECOIN"],
        "CRYPTO_GEN": ["CRYPTO", "BITCOIN", "BTC", "SEC", "ETF", "BINANCE"] # تاثیر عمومی روی کل رمزارزها
    }
    
    # کلمات بحرانی (اقتصادی، جنگ و کریپتو)
    crisis_categories = {
        "WAR_GEOPOLITIC": ['WAR', 'MISSILE', 'ATTACK', 'MILITARY', 'CEASEFIRE', 'INVASION', 'ESCALATION'],
        "ECONOMIC_CRASH": ['CRASH', 'BLACK SWAN', 'EMERGENCY RATE CUT', 'COLLAPSE', 'BANKRUPTCY', 'LIQUIDITY CRISIS'],
        "CRYPTO_CRISIS":  ['HACK', 'EXPLOIT', 'SEC LAWSUIT', 'RUG PULL', 'MT GOX', 'FTX']
    }
    
    crisis_status = {c: {"active": False, "type": "none"} for c in asset_keywords}
    sentiment_scores = {c: [] for c in asset_keywords}
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200: 
                continue
                
            root = ET.fromstring(response.content)
            
            # بررسی 40 خبر مهم از هر فید
            for item in root.findall('.//item')[:40]: 
                title = (item.find('title').text or "")
                desc = (item.find('description').text or "")
                
                full_text = f"{title}. {desc}"
                upper_text = full_text.upper()
                
                # استخراج امتیاز احساسات توسط هوش مصنوعی FinBERT
                score = 0.0
                if finbert_nlp:
                    try:
                        # محدودیت 512 توکن برای جلوگیری از کرش کردن مدل
                        nlp_result = finbert_nlp(full_text[:500])[0] 
                        if nlp_result['label'] == 'positive':
                            score = nlp_result['score']
                        elif nlp_result['label'] == 'negative':
                            score = -nlp_result['score']
                    except:
                        pass
                
                # تخصیص امتیاز به دارایی‌های مرتبط
                for asset, keywords in asset_keywords.items():
                    if any(kw in upper_text for kw in keywords):
                        sentiment_scores[asset].append(score)
                        
                        # بررسی فعال‌سازی حالت بحران
                        for c_type, c_words in crisis_categories.items():
                            if any(cw in upper_text for cw in c_words):
                                crisis_status[asset] = {"active": True, "type": c_type}
                                
        except Exception as e: 
            print(f"[WARNING] NLP Feed parsing error for URL {url[:30]}... : {e}")
            pass

    # ساخت ماتریس نهایی
    final_matrix = {}
    for asset in asset_keywords.keys():
        avg_score = np.mean(sentiment_scores[asset]) if len(sentiment_scores[asset]) > 0 else 0.0
        final_matrix[asset] = {
            "sentiment": round(float(avg_score), 3),
            "crisis_mode": crisis_status[asset]["active"]
        }
        
    return final_matrix

# ======================================================================
# 5. INSTITUTIONAL ORDER BLOCK ENGINE (SMC)
# ======================================================================

def detect_institutional_order_blocks(df, lookback=50):
    """
    استخراج دقیق محدوده‌های معاملاتی (Dealing Ranges) و بلوک‌های سفارش بانک‌ها.
    """
    if len(df) < lookback: 
        return None
        
    recent = df.iloc[-lookback:-1].copy()
    
    dealing_high = recent['High'].max()
    dealing_low = recent['Low'].min()
    equilibrium = (dealing_high + dealing_low) / 2.0
    
    order_blocks = {
        "bullish_ob_top": 0.0, "bullish_ob_bottom": 0.0, "bullish_valid": False,
        "bearish_ob_top": 0.0, "bearish_ob_bottom": 0.0, "bearish_valid": False,
        "equilibrium": equilibrium, "dealing_high": dealing_high, "dealing_low": dealing_low
    }
    
    # جستجو برای اوردر بلاک صعودی (در منطقه ارزان‌فروشی)
    for i in range(len(recent) - 5, 2, -1):
        cond_down_candle = recent['Close'].iloc[i] < recent['Open'].iloc[i]
        cond_strong_move = (recent['Close'].iloc[i+1] > recent['High'].iloc[i] and 
                            recent['Close'].iloc[i+2] > recent['Close'].iloc[i+1])
        
        if cond_down_candle and cond_strong_move:
            if recent['High'].iloc[i] < equilibrium:
                order_blocks["bullish_ob_top"] = recent['High'].iloc[i]
                order_blocks["bullish_ob_bottom"] = recent['Low'].iloc[i]
                order_blocks["bullish_valid"] = True
                break
                
    # جستجو برای اوردر بلاک نزولی (در منطقه گران‌فروشی)
    for i in range(len(recent) - 5, 2, -1):
        cond_up_candle = recent['Close'].iloc[i] > recent['Open'].iloc[i]
        cond_sharp_drop = (recent['Close'].iloc[i+1] < recent['Low'].iloc[i] and 
                           recent['Close'].iloc[i+2] < recent['Close'].iloc[i+1])
        
        if cond_up_candle and cond_sharp_drop:
            if recent['Low'].iloc[i] > equilibrium:
                order_blocks["bearish_ob_top"] = recent['High'].iloc[i]
                order_blocks["bearish_ob_bottom"] = recent['Low'].iloc[i]
                order_blocks["bearish_valid"] = True
                break
                
    return order_blocks

# ======================================================================
# 6. XGBOOST AI ENGINE (Machine Learning)
# ======================================================================

def run_ml_prediction_for_asset(df):
    """
    آموزش زنده هوش مصنوعی (XGBoost) روی هر دارایی بر اساس دیتای تکنیکال.
    """
    df_ml = df.copy()
    
    df_ml['rsi'] = calculate_rsi(df_ml['Close'])
    df_ml['bb_width'] = calculate_bollinger_width(df_ml)
    vwap_line = calculate_vwap(df_ml)
    df_ml['vwap_dist'] = df_ml['Close'] - vwap_line
    df_ml['returns'] = df_ml['Close'].pct_change()
    
    df_ml['target'] = np.where(df_ml['Close'].shift(-1) > df_ml['Close'], 1, 0)
    df_ml.dropna(inplace=True)
    
    X = df_ml[['rsi', 'bb_width', 'vwap_dist', 'returns']]
    y = df_ml['target']
    
    if len(X) < 100: 
        return 0.5
    
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05, 
        random_state=42, eval_metric='logloss'
    )
    
    model.fit(X.iloc[:-1], y.iloc[:-1])
    prediction_probability = model.predict_proba(X.iloc[[-1]])[0][1]
    
    return prediction_probability

# ======================================================================
# 7. QUANTUM ENGINE V14: MAIN STRATEGY & EXECUTION
# ======================================================================

def generate_god_mode_strategy():
    run_time_utc = datetime.datetime.utcnow()
    print(f"\n[START] Quantum Engine V14 - {run_time_utc.strftime('%Y-%m-%d %H:%M UTC')}")
    
    news_matrix = analyze_crisis_news()
    corr_matrix = calculate_portfolio_correlation(SYMBOLS)
    
    portfolio_results = {}
    active_positions = [] 
    
    for symbol in SYMBOLS:
        print(f"[INFO] Deep Scanning: {symbol}")
        session_active = is_market_liquid_now(symbol)
        
        df_m15 = yf.download(symbol, period="15d", interval="15m", progress=False)
        df_h1 = yf.download(symbol, period="30d", interval="1h", progress=False)
        
        if df_m15.empty or df_h1.empty: 
            continue
            
        df_m15.columns = [col[0] if isinstance(col, tuple) else col for col in df_m15.columns]
        df_h1.columns = [col[0] if isinstance(col, tuple) else col for col in df_h1.columns]
        
        # تشخیص روند ماکرو و میکرو
        df_h4 = df_h1.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        df_h4['ema_50'] = df_h4['Close'].ewm(span=50, adjust=False).mean()
        macro_trend = "buy" if df_h4['Close'].iloc[-1] > df_h4['ema_50'].iloc[-1] else "sell"
        
        df_h1['ema_50'] = df_h1['Close'].ewm(span=50, adjust=False).mean()
        micro_trend = "buy" if df_h1['Close'].iloc[-1] > df_h1['ema_50'].iloc[-1] else "sell"
        
        target_trend = macro_trend if (macro_trend == micro_trend) else "flat"
        
                df_m15['vwap'] = calculate_vwap(df_m15)
        df_m15['bb_width'] = calculate_bollinger_width(df_m15)
        poc_current = calculate_poc(df_m15, lookback=100)
        
        # ======================================================================
        # سیستم هوشمند تشخیص ضریب پیپ (پشتیبانی از تمام کراس‌ها)
        # ======================================================================
        is_jpy_pair = "JPY" in symbol
        if is_jpy_pair:
            pip_multiplier = 100
        elif symbol in ["GC=F", "ETH-USD"]:
            pip_multiplier = 10     
        elif symbol == "LTC-USD":
            pip_multiplier = 100    
        elif symbol == "DOGE-USD":
            pip_multiplier = 10000  
        else:
            pip_multiplier = 10000  
            
        tr = pd.concat([
            df_m15['High'] - df_m15['Low'], 
            np.abs(df_m15['High'] - df_m15['Close'].shift()), 
            np.abs(df_m15['Low'] - df_m15['Close'].shift())
        ], axis=1).max(axis=1)
        df_m15['atr'] = tr.rolling(14).mean()

        current = df_m15.iloc[-2] 
        hurst = calculate_hurst(df_m15['Close'].tail(100).values)
        market_regime = "trending" if hurst > 0.52 else "range"
        
        liquidity_sweep = check_liquidity_sweep(df_m15)
        fvgs = detect_fair_value_gaps(df_m15)
        ob_data = detect_institutional_order_blocks(df_m15)
        vsa_anomaly = detect_vsa_anomaly(df_m15) 
        is_squeezing = detect_volatility_squeeze(df_m15) # ⚡️ قدرت جدید
        
        if not ob_data: 
            continue
        
        ml_prob_up = run_ml_prediction_for_asset(df_m15)
        
        direction = "flat"
        atr_pips = (current['atr'] * pip_multiplier)
        price_current = current['Close']
        bb_width_current = current['bb_width']
        current_vwap = current['vwap']
        current_rsi = calculate_rsi(df_m15['Close']).iloc[-2]
        
        # استخراج ارز پایه و مظنه 
        clean_sym = symbol.replace("=X", "").replace("-", "")
        if symbol == "GC=F":
            base_curr, quote_curr = "XAU", "USD"
        elif len(clean_sym) >= 6:
            base_curr = clean_sym[:3]
            quote_curr = clean_sym[3:6]
        else:
            base_curr, quote_curr = clean_sym, "USD"
            
        base_news = news_matrix.get(base_curr, {"sentiment": 0.0, "crisis_mode": False})
        quote_news = news_matrix.get(quote_curr, {"sentiment": 0.0, "crisis_mode": False})
        crypto_gen_news = news_matrix.get("CRYPTO_GEN", {"sentiment": 0.0, "crisis_mode": False})
        
        is_crisis = base_news["crisis_mode"] or quote_news["crisis_mode"]
        if base_curr in ["ETH", "LTC", "DOGE"] and crypto_gen_news["crisis_mode"]:
            is_crisis = True

        testing_bullish_ob = ob_data["bullish_valid"] and (ob_data["bullish_ob_bottom"] <= price_current <= ob_data["bullish_ob_top"])
        testing_bearish_ob = ob_data["bearish_valid"] and (ob_data["bearish_ob_bottom"] <= price_current <= ob_data["bearish_ob_top"])

        sl_pips = round(atr_pips * 1.5, 1)
        entry_zone_min = 0.0
        entry_zone_max = 0.0

        if is_crisis:
            if liquidity_sweep == "bullish_sweep" and target_trend != "sell":
                direction = "buy"
                sl_pips = round(atr_pips * 2.5, 1) 
                entry_zone_min = price_current - (atr_pips * 0.5 / pip_multiplier)
                entry_zone_max = price_current + (atr_pips * 0.5 / pip_multiplier)
            elif liquidity_sweep == "bearish_sweep" and target_trend != "buy":
                direction = "sell"
                sl_pips = round(atr_pips * 2.5, 1)
                entry_zone_min = price_current - (atr_pips * 0.5 / pip_multiplier)
                entry_zone_max = price_current + (atr_pips * 0.5 / pip_multiplier)
        else:
            if target_trend == "buy" and testing_bullish_ob and len(fvgs["bullish"]) > 0:
                direction = "buy"
                sl_price = min(ob_data["bullish_ob_bottom"], ob_data["dealing_low"])
                sl_pips = max(abs(price_current - sl_price) * pip_multiplier + 2.0, 10.0)
                entry_zone_min = round(ob_data["bullish_ob_bottom"], 5)
                entry_zone_max = round(ob_data["bullish_ob_top"], 5)
                
            elif target_trend == "sell" and testing_bearish_ob and len(fvgs["bearish"]) > 0:
                direction = "sell"
                sl_price = max(ob_data["bearish_ob_top"], ob_data["dealing_high"])
                sl_pips = max(abs(sl_price - price_current) * pip_multiplier + 2.0, 10.0)
                entry_zone_min = round(ob_data["bearish_ob_bottom"], 5)
                entry_zone_max = round(ob_data["bearish_ob_top"], 5)

        dynamic_threshold = 0.52 + min(bb_width_current * 10, 0.05)
        veto_reason = ""
        
        if direction != "flat":
            if not session_active: 
                direction = "flat"
                veto_reason = "Out of Session"
            elif direction == "buy" and current_rsi > 75.0:
                direction = "flat"
                veto_reason = "Overbought Exhaustion"
            elif direction == "sell" and current_rsi < 25.0:
                direction = "flat"
                veto_reason = "Oversold Exhaustion"
            elif not is_crisis:
                if direction == "buy" and ml_prob_up < dynamic_threshold: 
                    direction = "flat"
                    veto_reason = "XGBoost Veto"
                elif direction == "sell" and ml_prob_up > (1 - dynamic_threshold): 
                    direction = "flat"
                    veto_reason = "XGBoost Veto"
                elif direction == "buy" and price_current > current_vwap: 
                    direction = "flat"
                    veto_reason = "Above VWAP"
                elif direction == "sell" and price_current < current_vwap: 
                    direction = "flat"
                    veto_reason = "Below VWAP"

        if direction != "flat" and not corr_matrix.empty:
            for active_sym in active_positions:
                if symbol in corr_matrix.columns and active_sym in corr_matrix.columns:
                    corr_val = corr_matrix.loc[symbol, active_sym]
                    if abs(corr_val) > 0.85: 
                        direction = "flat"
                        veto_reason = f"Correlation Clash with {active_sym}"
                        break
        
        if direction != "flat": 
            active_positions.append(symbol)

        dist_to_high = abs(ob_data["dealing_high"] - price_current) * pip_multiplier
        dist_to_low = abs(price_current - ob_data["dealing_low"]) * pip_multiplier
        dist_to_poc = abs(price_current - poc_current) * pip_multiplier
        
        if direction != "flat":
            if is_crisis:
                tp1_pips = round(sl_pips * 1.5, 1)
                tp2_pips = round(sl_pips * 5.0, 1) 
            else:
                tp1_pips = round(max(dist_to_poc, sl_pips * 1.5), 1) 
                tp2_pips = round(max(dist_to_high if direction == "buy" else dist_to_low, tp1_pips * 2), 1)
            
            win_prob = ml_prob_up if direction == "buy" else (1 - ml_prob_up)
            active_risk = calculate_kelly_criterion(win_prob, tp1_pips, sl_pips)
            
            # ضریب ریسک تصاعدی: اگر نهنگ‌ها (VSA) وارد شده‌اند + بازار در حالت فشردگی (Squeeze) است
            if vsa_anomaly and is_squeezing: 
                active_risk = round(min(active_risk * 1.5, 3.0), 2)
            elif vsa_anomaly:
                active_risk = round(min(active_risk * 1.2, 3.0), 2)
        else:
            tp1_pips = round(atr_pips * 2.0, 1)
            tp2_pips = round(max(dist_to_high, dist_to_low, dist_to_poc), 1)
            active_risk = DEFAULT_RISK_PERCENT
            if target_trend == "buy":
                entry_zone_min = round(ob_data["bullish_ob_bottom"], 5) if ob_data["bullish_valid"] else round(price_current, 5)
                entry_zone_max = round(ob_data["bullish_ob_top"], 5) if ob_data["bullish_valid"] else round(price_current, 5)
            else:
                entry_zone_min = round(ob_data["bearish_ob_bottom"], 5) if ob_data["bearish_valid"] else round(price_current, 5)
                entry_zone_max = round(ob_data["bearish_ob_top"], 5) if ob_data["bearish_valid"] else round(price_current, 5)

        signal_expiry_str = (run_time_utc + timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')
        
        # ⚡️ سپر ضد-تکرار: ساخت هش امنیتی بی‌نقص برای قفل کردن اکسپرت متاتریدر
        sig_id = generate_signal_id(symbol, direction, signal_expiry_str) if direction != "flat" else "N/A"

        snapshot = {
            "analysis_price": round(price_current, 5),
            "vwap": round(current_vwap, 5),
            "rsi": round(current_rsi, 2),
            "atr_pips": round(atr_pips, 1),
            "macro_trend": target_trend,
            "market_regime": market_regime
        }

        portfolio_results[symbol] = {
            "signal_id": sig_id,
            "execution_lock": True if direction != "flat" else False, # برای اکسپرت: اطمینان از یکبار اجرا
            "regime": market_regime,
            "direction": direction,
            "entry_zone_min": entry_zone_min,
            "entry_zone_max": entry_zone_max,
            "market_state_snapshot": snapshot,
            "expiration_utc": signal_expiry_str,    
            "target_tp1_pips": round(max(tp1_pips, 5.0), 1),
            "target_tp2_pips": round(max(tp2_pips, 10.0), 1),
            "target_sl_pips": round(max(sl_pips, 3.0), 1),
            "risk_percent": active_risk,            
            "is_crisis": is_crisis,
            "vsa_confirmed": bool(vsa_anomaly),
            "squeeze_breakout": bool(is_squeezing), # وضعیت فشردگی برای گزارش تلگرام
            "veto": veto_reason
        }


    # ======================================================================
    # 8. MASTER JSON EXPORT & TELEGRAM REPORTING
    # ======================================================================
    output_data = {
        "last_update": run_time_utc.strftime('%Y-%m-%d %H:%M:%S'),
        "global_strategy": "Quantum_Engine_V14",
        "assets": portfolio_results
    }
    
    try:
        with open("sentinel_config.json", "w") as f: 
            json.dump(output_data, f, indent=4)
        print("[SUCCESS] sentinel_config.json saved successfully.")
    except Exception as e: 
        print(f"[ERROR] JSON Save failed: {e}")
        
    msg = f"<b>🌍 Quantum Engine V14 Live Feed</b>\n"
    msg += f"⏱ {run_time_utc.strftime('%H:%M UTC')}\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
    
    for sym, data in portfolio_results.items():
        clean_sym = sym.replace("=X", "") 
        mode_icon = "🚨 CRISIS" if data.get('is_crisis', False) else "🛡 SMC"
        vol_icon = "🔥 VSA" if data.get('vsa_confirmed', False) else "📊"
        
        if data['direction'] != "flat":
            msg += f"🟢 <b>{clean_sym}</b>: <b>{data['direction'].upper()}</b> [{mode_icon}]\n"
            msg += f"   🎯 Zone: {data['entry_zone_min']} ↔️ {data['entry_zone_max']} [{vol_icon}]\n"
            msg += f"   🎯 TP1: {data['target_tp1_pips']} | TP2: {data['target_tp2_pips']}\n"
            msg += f"   🛑 SL: {data['target_sl_pips']} | ⚖️ Risk: {data['risk_percent']}%\n"
            msg += f"   🆔 ID: <code>{data['signal_id']}</code>\n"
        else:
            reason = data['veto'] if data['veto'] else "Monitoring..."
            msg += f"⚪️ <b>{clean_sym}</b>: FLAT [{mode_icon}]\n"
            msg += f"   ⏳ <i>{reason}</i>\n"
            
        msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
            
    send_telegram_alert(msg)
    print("\n[SUCCESS] Quantum Engine V14 Execution Completed.")

if __name__ == "__main__":
    generate_god_mode_strategy()
