import os
import json
import datetime
from datetime import timedelta
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import xgboost as xgb
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# دانلود دیتابیس کلمات پردازش زبان طبیعی
nltk.download('vader_lexicon', quiet=True)

# ======================================================================
# 1. INSTITUTIONAL CONFIGURATION & TIME CORES
# ======================================================================

TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE" 
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"     

# سبد جامع دارایی‌های جهانی
SYMBOLS = [
    "EURUSD=X", 
    "GBPUSD=X", 
    "AUDUSD=X", 
    "USDCAD=X", 
    "USDJPY=X",
    "EURJPY=X", 
    "GBPAUD=X", 
    "EURGBP=X", 
    "AUDNZD=X", 
    "CHFJPY=X"
]

# ریسک پیش‌فرض
DEFAULT_RISK_PERCENT = 1.0 

# ساعات طلایی نقدینگی (سشن لندن و نیویورک به وقت UTC)
TRADING_HOURS = {
    "london_start": 7, 
    "new_york_end": 21
}

def is_market_liquid_now():
    """
    بررسی اینکه آیا بازار در زمان اوج حجم و نقدینگی قرار دارد یا خیر.
    """
    current_hour = datetime.datetime.utcnow().hour
    current_day = datetime.datetime.utcnow().weekday()
    
    # تعطیلات آخر هفته
    if current_day >= 5: 
        return False
        
    if TRADING_HOURS["london_start"] <= current_hour <= TRADING_HOURS["new_york_end"]:
        return True
    else:
        return False

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
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        print(f"[ERROR] Telegram alert failed: {e}")

# ======================================================================
# 2. ADVANCED RISK MANAGEMENT (Kelly Criterion & Correlation)
# ======================================================================

def calculate_kelly_criterion(win_probability, tp_pips, sl_pips):
    """
    محاسبه حجم بهینه با فرمول Half-Kelly Criterion.
    """
    if sl_pips == 0 or win_probability < 0.50: 
        return 0.0
        
    b = tp_pips / sl_pips
    p = win_probability
    q = 1.0 - p
    
    kelly_fraction = p - (q / b)
    half_kelly_percent = (kelly_fraction / 2.0) * 100
    
    final_risk = max(min(half_kelly_percent, 3.0), 0.5)
    return round(final_risk, 2)

def calculate_portfolio_correlation(symbols_list, lookback_days=10):
    """
    محاسبه ماتریس همبستگی پیرسون برای جلوگیری از ورود همزمان به ارزهای مشابه.
    """
    print("[INFO] Computing Live Pearson Correlation Matrix...")
    close_prices = {}
    
    for sym in symbols_list:
        try:
            df = yf.download(sym, period=f"{lookback_days}d", interval="1h", progress=False)
            if not df.empty:
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                close_prices[sym] = df['Close']
        except Exception as e: 
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
        # بررسی Bullish FVG
        if recent['Low'].iloc[i+1] > recent['High'].iloc[i-1]:
            if recent['Close'].iloc[i] > recent['Open'].iloc[i]:
                fvgs["bullish"].append((recent['High'].iloc[i-1], recent['Low'].iloc[i+1]))
                
        # بررسی Bearish FVG
        if recent['High'].iloc[i+1] < recent['Low'].iloc[i-1]:
            if recent['Close'].iloc[i] < recent['Open'].iloc[i]:
                fvgs["bearish"].append((recent['Low'].iloc[i-1], recent['High'].iloc[i+1]))
                
    return fvgs

# ======================================================================
# 4. GLOBAL CRISIS NLP ENGINE (موتور پردازش فاندامنتال کلان)
# ======================================================================

def analyze_crisis_news():
    """
    تحلیل زبان طبیعی (NLP) اخبار لحظه‌ای برای 8 ارز پایه جهان.
    تشخیص وضعیت بحران‌های ژئوپلیتیک (جنگ) و بحران‌های اقتصادی (سقوط بازار).
    """
    print("[INFO] Initializing V12 NLP Matrix...")
    sia = SentimentIntensityAnalyzer()
    
    urls = [
        "https://www.fxstreet.com/rss/news", 
        "https://www.forexlive.com/feed/news"
    ]
    
    country_keywords = {
        "USD": ["FED", "FOMC", "CPI", "NFP", "POWELL", "US ", "TREASURY", "DOLLAR"],
        "EUR": ["ECB", "LAGARDE", "EUROZONE", "GERMANY", "FRANCE"],
        "GBP": ["BOE", "BAILEY", "UK ", "BRITAIN", "BREXIT"],
        "AUD": ["RBA", "AUSTRALIA", "AUSSIE", "SYDNEY"],
        "CAD": ["BOC", "CANADA", "LOONIE", "OTTAWA", "OIL"],
        "JPY": ["BOJ", "YEN", "JAPAN", "TOKYO", "UEDA"],
        "NZD": ["RBNZ", "NEW ZEALAND", "KIWI", "WELLINGTON"],
        "CHF": ["SNB", "SWISS", "FRANC", "SWITZERLAND", "ZURICH"]
    }
    
    crisis_categories = {
        "WAR_GEOPOLITIC": ['WAR', 'MISSILE', 'ATTACK', 'MILITARY', 'CEASEFIRE', 'INVASION', 'IRAN', 'ISRAEL', 'RUSSIA', 'UKRAINE'],
        "ECONOMIC_CRASH": ['CRASH', 'BLACK SWAN', 'EMERGENCY RATE CUT', 'COLLAPSE', 'BANKRUPTCY', 'LIQUIDITY CRISIS']
    }
    
    crisis_status = {c: {"active": False, "type": "none"} for c in country_keywords}
    sentiment_scores = {c: [] for c in country_keywords}
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200: 
                continue
                
            root = ET.fromstring(response.content)
            
            # جستجو در 50 خبر داغ اخیر
            for item in root.findall('.//item')[:50]: 
                title = (item.find('title').text or "")
                desc = (item.find('description').text or "")
                
                full_text = f"{title}. {desc}".upper()
                score = sia.polarity_scores(f"{title}. {desc}")['compound']
                
                for curr, keywords in country_keywords.items():
                    if any(kw in full_text for kw in keywords):
                        sentiment_scores[curr].append(score)
                        
                        # بررسی کلمات بحرانی برای فعال‌سازی حالت Crisis Alpha
                        for c_type, c_words in crisis_categories.items():
                            if any(cw in full_text for cw in c_words):
                                crisis_status[curr] = {"active": True, "type": c_type}
        except Exception as e: 
            print(f"[WARNING] RSS Feed parsing error: {e}")
            pass

    # ساخت ماتریس نهایی و میانگین‌گیری از احساسات
    final_matrix = {}
    for curr in country_keywords.keys():
        avg_score = np.mean(sentiment_scores[curr]) if len(sentiment_scores[curr]) > 0 else 0.0
        final_matrix[curr] = {
            "sentiment": round(avg_score, 3),
            "crisis_mode": crisis_status[curr]["active"]
        }
        
    return final_matrix

# ======================================================================
# 5. INSTITUTIONAL ORDER BLOCK ENGINE (موتور کشف اوردر بلاک و نقدینگی)
# ======================================================================

def detect_institutional_order_blocks(df, lookback=50):
    """
    استخراج دقیق محدوده‌های معاملاتی (Dealing Ranges) و بلوک‌های سفارش بانک‌ها.
    """
    if len(df) < lookback: 
        return None
        
    recent = df.iloc[-lookback:-1].copy()
    
    # تعیین محدوده معاملاتی زنده و نقطه تعادل (Equilibrium)
    dealing_high = recent['High'].max()
    dealing_low = recent['Low'].min()
    equilibrium = (dealing_high + dealing_low) / 2.0
    
    order_blocks = {
        "bullish_ob_top": 0.0, 
        "bullish_ob_bottom": 0.0, 
        "bullish_valid": False,
        "bearish_ob_top": 0.0, 
        "bearish_ob_bottom": 0.0, 
        "bearish_valid": False,
        "equilibrium": equilibrium, 
        "dealing_high": dealing_high, 
        "dealing_low": dealing_low
    }
    
    # جستجو برای اوردر بلاک صعودی (در منطقه ارزان‌فروشی / Discount Zone)
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
                
    # جستجو برای اوردر بلاک نزولی (در منطقه گران‌فروشی / Premium Zone)
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
# 6. XGBOOST AI ENGINE (موتور هوش مصنوعی سازمانی)
# ======================================================================

def run_ml_prediction_for_asset(df):
    """
    آموزش زنده هوش مصنوعی Extreme Gradient Boosting (XGBoost) روی هر دارایی.
    """
    df_ml = df.copy()
    
    df_ml['rsi'] = calculate_rsi(df_ml['Close'])
    df_ml['bb_width'] = calculate_bollinger_width(df_ml)
    vwap_line = calculate_vwap(df_ml)
    df_ml['vwap_dist'] = df_ml['Close'] - vwap_line
    df_ml['returns'] = df_ml['Close'].pct_change()
    
    # آیا کندل بعدی صعودی خواهد بود؟ (1=بله، 0=خیر)
    df_ml['target'] = np.where(df_ml['Close'].shift(-1) > df_ml['Close'], 1, 0)
    
    df_ml.dropna(inplace=True)
    
    X = df_ml[['rsi', 'bb_width', 'vwap_dist', 'returns']]
    y = df_ml['target']
    
    if len(X) < 100: 
        return 0.5
    
    # پیکربندی مدل XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.05, 
        random_state=42, 
        eval_metric='logloss'
    )
    
    model.fit(X.iloc[:-1], y.iloc[:-1])
    prediction_probability = model.predict_proba(X.iloc[[-1]])[0][1]
    
    return prediction_probability


# ======================================================================
# 7. QUANTUM ENGINE: MAIN STRATEGY & EXECUTION (موتور اصلی و اجرای استراتژی)
# ======================================================================

def generate_god_mode_strategy():
    """
    موتور اصلی سیستم (Quantum Engine V12).
    شامل رفع باگ محاسباتی JPY، جایگزینی مناطق 0.0، و فیلتر خستگی بازار (RSI Exhaustion).
    """
    run_time_utc = datetime.datetime.utcnow()
    print(f"\n[START] Quantum Engine V12 - {run_time_utc.strftime('%Y-%m-%d %H:%M UTC')}")
    
    session_active = is_market_liquid_now()
    news_matrix = analyze_crisis_news()
    corr_matrix = calculate_portfolio_correlation(SYMBOLS)
    
    portfolio_results = {}
    active_positions = [] 
    
    for symbol in SYMBOLS:
        print(f"[INFO] Deep Scanning: {symbol}")
        
        df_m15 = yf.download(symbol, period="15d", interval="15m", progress=False)
        df_h1 = yf.download(symbol, period="30d", interval="1h", progress=False)
        
        if df_m15.empty or df_h1.empty: 
            continue
            
        df_m15.columns = [col[0] if isinstance(col, tuple) else col for col in df_m15.columns]
        df_h1.columns = [col[0] if isinstance(col, tuple) else col for col in df_h1.columns]
        
        df_h4 = df_h1.resample('4h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        df_h4['ema_50'] = df_h4['Close'].ewm(span=50, adjust=False).mean()
        macro_trend = "buy" if df_h4['Close'].iloc[-1] > df_h4['ema_50'].iloc[-1] else "sell"
        
        df_h1['ema_50'] = df_h1['Close'].ewm(span=50, adjust=False).mean()
        micro_trend = "buy" if df_h1['Close'].iloc[-1] > df_h1['ema_50'].iloc[-1] else "sell"
        
        target_trend = macro_trend if (macro_trend == micro_trend) else "flat"
        
        df_m15['vwap'] = calculate_vwap(df_m15)
        df_m15['bb_width'] = calculate_bollinger_width(df_m15)
        poc_current = calculate_poc(df_m15, lookback=100)
        
        # ======================================================================
        # حل مشکل محاسبه پیپ برای ین ژاپن (JPY)
        # ======================================================================
        pip_multiplier = 100 if "JPY" in symbol else 10000
        
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
        
        if not ob_data: 
            continue
        
        ml_prob_up = run_ml_prediction_for_asset(df_m15)
        
        direction = "flat"
        # استفاده از ضریب صحیح برای محاسبه پیپ ATR
        atr_pips = (current['atr'] * pip_multiplier)
        price_current = current['Close']
        bb_width_current = current['bb_width']
        current_vwap = current['vwap']
        current_rsi = calculate_rsi(df_m15['Close']).iloc[-2]
        
        base_curr = symbol[:3]
        quote_curr = symbol[3:6]
        
        base_news = news_matrix.get(base_curr, {"sentiment": 0.0, "crisis_mode": False})
        quote_news = news_matrix.get(quote_curr, {"sentiment": 0.0, "crisis_mode": False})
        
        is_crisis = base_news["crisis_mode"] or quote_news["crisis_mode"]

        testing_bullish_ob = ob_data["bullish_valid"] and (ob_data["bullish_ob_bottom"] <= price_current <= ob_data["bullish_ob_top"])
        testing_bearish_ob = ob_data["bearish_valid"] and (ob_data["bearish_ob_bottom"] <= price_current <= ob_data["bearish_ob_top"])

        sl_pips = round(atr_pips * 1.5, 1)
        
        entry_zone_min = 0.0
        entry_zone_max = 0.0

        # ======================================================================
        # ماتریس ورود (تخصیص Entry Zone)
        # ======================================================================
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

        # ======================================================================
        # ماتریس فیلترها و وتو (Risk & Veto Matrix)
        # ======================================================================
        dynamic_threshold = 0.52 + min(bb_width_current * 10, 0.05)
        veto_reason = ""
        
        if direction != "flat":
            if not session_active: 
                direction = "flat"
                veto_reason = "Out of Session"
                
            # فیلتر جدید: جلوگیری از ورود در نقاط خستگی و اشباع بازار (RSI Exhaustion)
            elif direction == "buy" and current_rsi > 75.0:
                direction = "flat"
                veto_reason = "Overbought Exhaustion (RSI > 75)"
            elif direction == "sell" and current_rsi < 25.0:
                direction = "flat"
                veto_reason = "Oversold Exhaustion (RSI < 25)"
                
            elif not is_crisis:
                if direction == "buy" and ml_prob_up < dynamic_threshold: 
                    direction = "flat"
                    veto_reason = f"XGBoost (<{dynamic_threshold:.2f})"
                elif direction == "sell" and ml_prob_up > (1 - dynamic_threshold): 
                    direction = "flat"
                    veto_reason = f"XGBoost (>{(1-dynamic_threshold):.2f})"
                elif direction == "buy" and price_current > current_vwap: 
                    direction = "flat"
                    veto_reason = "Above VWAP"
                elif direction == "sell" and price_current < current_vwap: 
                    direction = "flat"
                    veto_reason = "Below VWAP"
            else:
                if direction == "buy" and ml_prob_up < 0.40: 
                    direction = "flat"
                    veto_reason = "Crisis ML Veto"
                elif direction == "sell" and ml_prob_up > 0.60: 
                    direction = "flat"
                    veto_reason = "Crisis ML Veto"

        # فیلتر همبستگی پیرسون
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

        # ======================================================================
        # سیستم تارگت‌ها، مدیریت ریسک Kelly و Market Snapshot
        # ======================================================================
        # جایگزینی ضریب 10000 با pip_multiplier برای محاسبه دقیق تمام ارزها
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
            
            if vsa_anomaly: 
                active_risk = round(min(active_risk * 1.2, 3.0), 2)
                
        else:
            # محاسبات تارگت در حالت FLAT
            tp1_pips = round(atr_pips * 2.0, 1)
            tp2_pips = round(max(dist_to_high, dist_to_low, dist_to_poc), 1)
            active_risk = DEFAULT_RISK_PERCENT
            
            # حل مشکل 0.0: اگر اوردر بلاک معتبری وجود داشت، محدوده آن را ثبت کن؛ در غیر این صورت قیمت فعلی
            if target_trend == "buy":
                entry_zone_min = round(ob_data["bullish_ob_bottom"], 5) if ob_data["bullish_valid"] else round(price_current, 5)
                entry_zone_max = round(ob_data["bullish_ob_top"], 5) if ob_data["bullish_valid"] else round(price_current, 5)
            else:
                entry_zone_min = round(ob_data["bearish_ob_bottom"], 5) if ob_data["bearish_valid"] else round(price_current, 5)
                entry_zone_max = round(ob_data["bearish_ob_top"], 5) if ob_data["bearish_valid"] else round(price_current, 5)

        signal_expiry_str = (run_time_utc + timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')

        snapshot = {
            "analysis_price": round(price_current, 5),
            "vwap": round(current_vwap, 5),
            "rsi": round(current_rsi, 2),
            "atr_pips": round(atr_pips, 1),
            "macro_trend": target_trend,
            "market_regime": market_regime
        }

        portfolio_results[symbol] = {
            "regime": market_regime,
            "direction": direction,
            "entry_zone_min": entry_zone_min,
            "entry_zone_max": entry_zone_max,
            "market_state_snapshot": snapshot,
            "expiration_utc": signal_expiry_str,    
            "target_tp1_pips": round(max(tp1_pips, 10.0), 1),
            "target_tp2_pips": round(max(tp2_pips, 20.0), 1),
            "target_sl_pips": round(max(sl_pips, 5.0), 1),
            "risk_percent": active_risk,            
            "is_crisis": is_crisis,
            "vsa_confirmed": bool(vsa_anomaly),     
            "veto": veto_reason
        }

    # ======================================================================
    # 8. MASTER JSON EXPORT & TELEGRAM REPORTING
    # ======================================================================
    output_data = {
        "last_update": run_time_utc.strftime('%Y-%m-%d %H:%M:%S'),
        "global_strategy": "Quantum_Engine_V12",
        "session_active": session_active,
        "assets": portfolio_results
    }
    
    try:
        with open("sentinel_config.json", "w") as f: 
            json.dump(output_data, f, indent=4)
        print("[SUCCESS] sentinel_config.json saved successfully.")
    except Exception as e: 
        print(f"[ERROR] JSON Save failed: {e}")
        
    msg = f"<b>🌍 Quantum Engine V12 Live Feed</b>\n"
    msg += f"⏱ {run_time_utc.strftime('%H:%M UTC')}\n"
    msg += f"🏢 Session: {'🟢 Active' if session_active else '🔴 Inactive'}\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
    
    for sym, data in portfolio_results.items():
        clean_sym = sym.replace("=X", "") 
        mode_icon = "🚨 CRISIS" if data.get('is_crisis', False) else "🛡 SMC+FVG"
        vol_icon = "🔥 VSA" if data.get('vsa_confirmed', False) else "📊"
        
        if data['direction'] != "flat":
            msg += f"🟢 <b>{clean_sym}</b>: <b>{data['direction'].upper()}</b> [{mode_icon}]\n"
            msg += f"   🎯 Zone: {data['entry_zone_min']} ↔️ {data['entry_zone_max']} [{vol_icon}]\n"
            msg += f"   🎯 TP1: {data['target_tp1_pips']} | TP2: {data['target_tp2_pips']}\n"
            msg += f"   🛑 SL: {data['target_sl_pips']} | ⚖️ Risk: {data['risk_percent']}%\n"
            msg += f"   ⏳ Expires: {data['expiration_utc']}\n"
        else:
            reason = data['veto'] if data['veto'] else "Monitoring OB & FVG"
            msg += f"⚪️ <b>{clean_sym}</b>: FLAT [{mode_icon}]\n"
            msg += f"   ⏳ <i>{reason}</i>\n"
            
        msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
            
    send_telegram_alert(msg)
    print("\n[SUCCESS] Quantum Engine V12 Execution Completed.")

if __name__ == "__main__":
    generate_god_mode_strategy()
