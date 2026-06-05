import os
import json
import datetime
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
# 1. INSTITUTIONAL CONFIGURATION & RISK CORES (تنظیمات کلان و هسته ریسک)
# ======================================================================
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE" 
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"     

# سبد جامع دارایی‌های ۱۰گانه جهانی
SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "USDJPY=X",
    "EURJPY=X", "GBPAUD=X", "EURGBP=X", "AUDNZD=X", "CHFJPY=X"
]

NORMAL_RISK_PERCENT = 2.0 
CRISIS_RISK_PERCENT = 1.0 

# مدیریت ساعات معاملاتی (به وقت UTC) - فیلتر نویز سشن‌های مرده
TRADING_HOURS = {
    "london_start": 7,    # ساعت 07:00 UTC شروع سشن لندن
    "new_york_end": 21   # ساعت 21:00 UTC پایان سشن نیویورک
}

def is_market_liquid_now():
    """ بررسی اینکه آیا بازار در زمان اوج حجم و نقدینگی بانک‌ها قرار دارد یا خیر """
    current_hour = datetime.datetime.utcnow().hour
    current_day = datetime.datetime.utcnow().weekday()
    
    if current_day >= 5: # تعطیلات آخر هفته بازار جهانی
        return False
    
    # اجازه ترید فقط در بازه حجم بالای لندن و نیویورک
    return TRADING_HOURS["london_start"] <= current_hour <= TRADING_HOURS["new_york_end"]

def send_telegram_alert(message):
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE": return 
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try: requests.post(url, data=payload, timeout=5)
    except: pass

# ======================================================================
# 2. PEARSON CORRELATION MATRIX (ماتریس همبستگی پیرسون برای مهار ریسک هم‌پوشانی)
# ======================================================================
def calculate_portfolio_correlation(symbols_list, lookback_days=10):
    """
    محاسبه زنده همبستگی دارایی‌ها برای جلوگیری از باز شدن پوزیشن‌های هم‌جهت تکراری.
    این سیستم مانع از مالتی‌پلی شدن ریسک روی یک ارز خاص (مثل ریسک سنگین روی دلار) می‌شود.
    """
    print("[INFO] Computing Live Pearson Correlation Matrix...")
    close_prices = {}
    
    for sym in symbols_list:
        try:
            df = yf.download(sym, period=f"{lookback_days}d", interval="1h", progress=False)
            if not df.empty:
                # یکسان‌سازی ستون‌ها برای جلوگیری از ارور مالتی ایندکس
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                close_prices[sym] = df['Close']
        except: pass
        
    if len(close_prices) < 2:
        return pd.DataFrame()
        
    df_corr = pd.DataFrame(close_prices).corr(method='pearson')
    print("[SUCCESS] Correlation Matrix Calculated.")
    return df_corr

# ======================================================================
# 3. QUANTITATIVE INDICATORS & SMART MONEY CONCEPTS (کوانت و پول هوشمند)
# ======================================================================
def calculate_vwap(df):
    """ محاسبه میانگین قیمت وزنی حجم (ارزش واقعی از دیدگاه بانک‌ها) """
    q = df['Volume'].replace(0, 1) 
    p = (df['High'] + df['Low'] + df['Close']) / 3
    return (p * q).cumsum() / q.cumsum()

def calculate_poc(df, lookback=100):
    """ محاسبه Point of Control (POC): نقطه‌ای که بیشترین حجم پول در آن درگیر است """
    if len(df) < lookback: return df['Close'].iloc[-1]
    recent = df.iloc[-lookback:].copy()
    if recent['Volume'].sum() == 0: return recent['Close'].mean()
    recent['Price_Bin'] = recent['Close'].round(4) # گروه‌بندی قیمت‌ها تا 4 رقم اعشار
    return recent.groupby('Price_Bin')['Volume'].sum().idxmax()

def calculate_hurst(price_series, max_lag=20):
    """ تشخیص رژیم بازار (رونددار یا خنثی) با فرمول Hurst Exponent """
    if len(price_series) < max_lag: return 0.5
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_width(df, window=20):
    """ محاسبه فشردگی بازار برای تشخیص زمان انفجارهای قیمتی """
    std = df['Close'].rolling(window=window).std()
    ma = df['Close'].rolling(window=window).mean()
    upper = ma + (std * 2)
    lower = ma - (std * 2)
    return (upper - lower) / ma

def check_liquidity_sweep(df, lookback=20):
    """ بررسی شکار شدن استاپ‌لاس خرده‌پاها توسط نهنگ‌ها """
    if len(df) < lookback + 2: return "none"
    recent_high = df['High'].iloc[-lookback-2:-2].max()
    recent_low = df['Low'].iloc[-lookback-2:-2].min()
    current = df.iloc[-2] 
    
    if current['High'] > recent_high and current['Close'] < recent_high:
        return "bearish_sweep"
    elif current['Low'] < recent_low and current['Close'] > recent_low:
        return "bullish_sweep"
    return "none"

def detect_fair_value_gaps(df, lookback=15):
    """
    کشف گپ‌های ارزش منصفانه (FVG) زنده در بازار
    اوردر بلاک‌ها فقط زمانی ارزش دارند که بانک‌ها یک گپ (عدم تعادل) پشت سر خود جا گذاشته باشند.
    """
    fvgs = {"bullish": [], "bearish": []}
    if len(df) < lookback: return fvgs
    
    recent = df.iloc[-lookback:-1]
    for i in range(1, len(recent)-1):
        # Bullish FVG: فاصله بین سقف کندل اول و کف کندل سوم
        if recent['Low'].iloc[i+1] > recent['High'].iloc[i-1]:
            if recent['Close'].iloc[i] > recent['Open'].iloc[i]: # کندل میانی صعودی باشد
                fvgs["bullish"].append((recent['High'].iloc[i-1], recent['Low'].iloc[i+1]))
                
        # Bearish FVG: فاصله بین کف کندل اول و سقف کندل سوم
        if recent['High'].iloc[i+1] < recent['Low'].iloc[i-1]:
            if recent['Close'].iloc[i] < recent['Open'].iloc[i]: # کندل میانی نزولی باشد
                fvgs["bearish"].append((recent['Low'].iloc[i-1], recent['High'].iloc[i+1]))
                
    return fvgs

# ======================================================================
# 4. GLOBAL CRISIS NLP ENGINE (موتور کلان پردازش فاندامنتال)
# ======================================================================
def analyze_crisis_news():
    """ تحلیل زبان طبیعی اخبار برای 10 اقتصاد بزرگ جهان و تشخیص بحران """
    print("[INFO] Initializing V11 Global NLP Matrix...")
    sia = SentimentIntensityAnalyzer()
    
    urls = [
        "https://www.fxstreet.com/rss/news",
        "https://www.forexlive.com/feed/news"
    ]
    
    country_keywords = {
        "USD": ["FED", "FOMC", "CPI", "NFP", "POWELL", "US ", "TREASURY", "DOLLAR"],
        "EUR": ["ECB", "LAGARDE", "EUROZONE", "GERMANY", "FRANCE", "EURO "],
        "GBP": ["BOE", "BAILEY", "UK ", "BRITAIN", "BREXIT", "STERLING"],
        "AUD": ["RBA", "AUSTRALIA", "AUSSIE", "SYDNEY"],
        "CAD": ["BOC", "CANADA", "LOONIE", "OTTAWA", "OIL"],
        "JPY": ["BOJ", "YEN", "JAPAN", "TOKYO", "UEDA"],
        "NZD": ["RBNZ", "NEW ZEALAND", "KIWI", "WELLINGTON"],
        "CHF": ["SNB", "SWISS", "FRANC", "SWITZERLAND", "ZURICH"]
    }
    
    crisis_categories = {
        "WAR_GEOPOLITIC": ['WAR', 'MISSILE', 'ATTACK', 'MILITARY', 'CEASEFIRE', 'INVASION', 'IRAN', 'ISRAEL', 'RUSSIA', 'UKRAINE', 'HEZBOLLAH'],
        "ECONOMIC_CRASH": ['CRASH', 'BLACK SWAN', 'EMERGENCY RATE CUT', 'COLLAPSE', 'BANKRUPTCY', 'LIQUIDITY CRISIS']
    }
    
    crisis_status = {currency: {"active": False, "type": "none"} for currency in country_keywords.keys()}
    sentiment_scores = {currency: [] for currency in country_keywords.keys()}
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200: continue
            root = ET.fromstring(response.content)
            
            # افزایش عمق جستجو به 50 خبر اخیر برای پوشش کامل 10 ارز
            for item in root.findall('.//item')[:50]: 
                title = (item.find('title').text or "")
                desc = (item.find('description').text or "")
                
                full_text_nlp = f"{title}. {desc}"
                full_text_upper = full_text_nlp.upper()
                
                compound_score = sia.polarity_scores(full_text_nlp)['compound']
                
                for currency, keywords in country_keywords.items():
                    if any(kw in full_text_upper for kw in keywords):
                        sentiment_scores[currency].append(compound_score)
                        
                        for c_type, c_words in crisis_categories.items():
                            if any(cw in full_text_upper for cw in c_words):
                                crisis_status[currency]["active"] = True
                                crisis_status[currency]["type"] = c_type
        except Exception as e:
            pass

    final_matrix = {}
    for curr in country_keywords.keys():
        avg_score = np.mean(sentiment_scores[curr]) if len(sentiment_scores[curr]) > 0 else 0.0
        final_matrix[curr] = {
            "sentiment": round(avg_score, 3),
            "crisis_mode": crisis_status[curr]["active"],
            "crisis_type": crisis_status[curr]["type"]
        }
    
    print("[SUCCESS] Crisis NLP Matrix Built Successfully.")
    return final_matrix 

# ======================================================================
# 5. INSTITUTIONAL ORDER BLOCK ENGINE (موتور کشف اوردر بلاک و نقدینگی)
# ======================================================================
def detect_institutional_order_blocks(df, lookback=50):
    """
    استخراج دقیق محدوده‌های معاملاتی (Dealing Ranges) و بلوک‌های سفارش بانک‌ها.
    این تابع فقط اوردر بلاک‌هایی را معتبر می‌داند که باعث شکست ساختار (BOS) شده باشند.
    """
    if len(df) < lookback: 
        return None
        
    recent = df.iloc[-lookback:-1].copy()
    
    # 1. تعیین محدوده معاملاتی زنده و نقطه تعادل (Equilibrium)
    dealing_high = recent['High'].max()
    dealing_low = recent['Low'].min()
    equilibrium = (dealing_high + dealing_low) / 2.0
    
    order_blocks = {
        "bullish_ob_top": 0.0, "bullish_ob_bottom": 0.0, "bullish_valid": False,
        "bearish_ob_top": 0.0, "bearish_ob_bottom": 0.0, "bearish_valid": False,
        "equilibrium": equilibrium, "dealing_high": dealing_high, "dealing_low": dealing_low
    }
    
    # 2. جستجو برای اوردر بلاک صعودی (در منطقه ارزان‌فروشی / Discount Zone)
    # به دنبال آخرین کندل نزولی قبل از یک رالی قدرتمند می‌گردیم
    for i in range(len(recent) - 5, 2, -1):
        cond_down_candle = recent['Close'].iloc[i] < recent['Open'].iloc[i]
        cond_strong_move = recent['Close'].iloc[i+1] > recent['High'].iloc[i] and recent['Close'].iloc[i+2] > recent['Close'].iloc[i+1]
        
        if cond_down_candle and cond_strong_move:
            if recent['High'].iloc[i] < equilibrium:
                order_blocks["bullish_ob_top"] = recent['High'].iloc[i]
                order_blocks["bullish_ob_bottom"] = recent['Low'].iloc[i]
                order_blocks["bullish_valid"] = True
                break
                
    # 3. جستجو برای اوردر بلاک نزولی (در منطقه گران‌فروشی / Premium Zone)
    # به دنبال آخرین کندل صعودی قبل از یک ریزش قدرتمند می‌گردیم
    for i in range(len(recent) - 5, 2, -1):
        cond_up_candle = recent['Close'].iloc[i] > recent['Open'].iloc[i]
        cond_sharp_drop = recent['Close'].iloc[i+1] < recent['Low'].iloc[i] and recent['Close'].iloc[i+2] < recent['Close'].iloc[i+1]
        
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
    این مدل خطای تصمیم‌گیری اندیکاتورهای کلاسیک را به حداقل می‌رساند.
    """
    df_ml = df.copy()
    
    # ساخت ویژگی‌ها (Feature Engineering)
    df_ml['rsi'] = calculate_rsi(df_ml['Close'])
    df_ml['bb_width'] = calculate_bollinger_width(df_ml)
    vwap_line = calculate_vwap(df_ml)
    df_ml['vwap_dist'] = df_ml['Close'] - vwap_line
    df_ml['returns'] = df_ml['Close'].pct_change()
    
    # تعریف هدف (Target): آیا کندل بعدی صعودی خواهد بود؟
    df_ml['target'] = np.where(df_ml['Close'].shift(-1) > df_ml['Close'], 1, 0)
    df_ml.dropna(inplace=True)
    
    X = df_ml[['rsi', 'bb_width', 'vwap_dist', 'returns']]
    y = df_ml['target']
    
    if len(X) < 100: 
        return 0.5
    
    # پیکربندی مدل XGBoost برای جلوگیری از نویز و بیش‌برازش
    model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.05, 
        random_state=42, 
        eval_metric='logloss'
    )
    
    # آموزش مدل روی داده‌های گذشته و پیش‌بینی کندل فعلی
    model.fit(X.iloc[:-1], y.iloc[:-1])
    return model.predict_proba(X.iloc[[-1]])[0][1]

# ======================================================================
# 7. TITAN ENGINE: MAIN STRATEGY & EXECUTION (موتور اصلی و اجرای استراتژی)
# ======================================================================
def generate_god_mode_strategy():
    print(f"\n[START] Titan Engine V11 - {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    
    # 1. بررسی نقدینگی سشن‌ها (آیا بانک‌ها در حال معامله هستند؟)
    session_active = is_market_liquid_now()
    print(f"[INFO] Session Active: {session_active}")

    # 2. استخراج داده‌های کلان
    news_matrix = analyze_crisis_news()
    corr_matrix = calculate_portfolio_correlation(SYMBOLS)
    
    portfolio_results = {}
    active_positions = [] # برای ردیابی دارایی‌های هم‌جهت و مهار همبستگی
    
    for symbol in SYMBOLS:
        print(f"[INFO] Deep Scanning: {symbol}")
        
        # دریافت داده‌ها از یاهو فایننس
        df_m15 = yf.download(symbol, period="15d", interval="15m", progress=False)
        df_h1 = yf.download(symbol, period="30d", interval="1h", progress=False)
        
        if df_m15.empty or df_h1.empty: 
            continue
            
        df_m15.columns = [col[0] if isinstance(col, tuple) else col for col in df_m15.columns]
        df_h1.columns = [col[0] if isinstance(col, tuple) else col for col in df_h1.columns]
        
        df_h4 = df_h1.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        
        # تحلیل روند چندگانه (MTF)
        df_h4['ema_50'] = df_h4['Close'].ewm(span=50, adjust=False).mean()
        macro_trend = "buy" if df_h4['Close'].iloc[-1] > df_h4['ema_50'].iloc[-1] else "sell"
        
        df_h1['ema_50'] = df_h1['Close'].ewm(span=50, adjust=False).mean()
        micro_trend = "buy" if df_h1['Close'].iloc[-1] > df_h1['ema_50'].iloc[-1] else "sell"
        
        target_trend = macro_trend if (macro_trend == micro_trend) else "flat"
        
        # محاسبات کوانت و اسمارت مانی
        df_m15['vwap'] = calculate_vwap(df_m15)
        df_m15['bb_width'] = calculate_bollinger_width(df_m15)
        poc_current = calculate_poc(df_m15, 100)
        
        tr = pd.concat([df_m15['High'] - df_m15['Low'], np.abs(df_m15['High'] - df_m15['Close'].shift()), np.abs(df_m15['Low'] - df_m15['Close'].shift())], axis=1).max(axis=1)
        df_m15['atr'] = tr.rolling(14).mean()

        current = df_m15.iloc[-2] 
        hurst = calculate_hurst(df_m15['Close'].tail(100).values)
        market_regime = "trending" if hurst > 0.52 else "range"
        
        liquidity_sweep = check_liquidity_sweep(df_m15)
        fvgs = detect_fair_value_gaps(df_m15)
        ob_data = detect_institutional_order_blocks(df_m15)
        
        if not ob_data: continue
        
        ml_prob_up = run_ml_prediction_for_asset(df_m15)
        
        direction = "flat"
        atr_pips = (current['atr'] * 10000)
        price_current = current['Close']
        bb_width_current = current['bb_width']
        
        # وضعیت فاندامنتال اختصاصی هر ارز
        base_curr = symbol[:3]
        quote_curr = symbol[3:6]
        base_news = news_matrix.get(base_curr, {"sentiment": 0.0, "crisis_mode": False, "crisis_type": "none"})
        quote_news = news_matrix.get(quote_curr, {"sentiment": 0.0, "crisis_mode": False, "crisis_type": "none"})
        
        is_crisis = base_news["crisis_mode"] or quote_news["crisis_mode"]
        active_risk = CRISIS_RISK_PERCENT if is_crisis else NORMAL_RISK_PERCENT

        # تاییدیه نفوذ به اوردر بلاک (Mitigation)
        testing_bullish_ob = ob_data["bullish_valid"] and (ob_data["bullish_ob_bottom"] <= price_current <= ob_data["bullish_ob_top"])
        testing_bearish_ob = ob_data["bearish_valid"] and (ob_data["bearish_ob_bottom"] <= price_current <= ob_data["bearish_ob_top"])

        sl_pips = round(atr_pips * 1.5, 1)

        # ======================================================================
        # ماتریس ورود (ترکیب بحران، اوردر بلاک و گپ‌های ارزش منصفانه FVG)
        # ======================================================================
        if is_crisis:
            # در بحران فقط استاپ‌هانت‌ها را ترید می‌کنیم
            if liquidity_sweep == "bullish_sweep" and target_trend != "sell":
                direction = "buy"
                sl_pips = round(atr_pips * 2.5, 1) 
            elif liquidity_sweep == "bearish_sweep" and target_trend != "buy":
                direction = "sell"
                sl_pips = round(atr_pips * 2.5, 1)
        else:
            # در روزهای عادی: اوردر بلاک + حضور حداقل یک FVG باز 
            if target_trend == "buy" and testing_bullish_ob and len(fvgs["bullish"]) > 0:
                direction = "buy"
                sl_price = min(ob_data["bullish_ob_bottom"], ob_data["dealing_low"])
                sl_pips = max(abs(price_current - sl_price) * 10000 + 2.0, 10.0) 
            elif target_trend == "sell" and testing_bearish_ob and len(fvgs["bearish"]) > 0:
                direction = "sell"
                sl_price = max(ob_data["bearish_ob_top"], ob_data["dealing_high"])
                sl_pips = max(abs(sl_price - price_current) * 10000 + 2.0, 10.0)

        # ======================================================================
        # ماتریس وتو فیلترها (مدیریت ریسک هوشمند)
        # ======================================================================
        dynamic_threshold = 0.52 + min(bb_width_current * 10, 0.05)
        veto_reason = ""
        
        if direction != "flat":
            # 1. فیلتر نقدینگی زمانی
            if not session_active:
                direction = "flat"
                veto_reason = "Out of Session (Low Liquidity)"
                
            # 2. فیلترهای روزهای عادی
            elif not is_crisis:
                if direction == "buy" and ml_prob_up < dynamic_threshold: 
                    direction = "flat"; veto_reason = f"XGBoost Veto (<{dynamic_threshold:.2f})"
                elif direction == "sell" and ml_prob_up > (1 - dynamic_threshold): 
                    direction = "flat"; veto_reason = f"XGBoost Veto (>{(1-dynamic_threshold):.2f})"
                elif direction == "buy" and price_current > current['vwap']: 
                    direction = "flat"; veto_reason = "Above VWAP (Not Discounted)"
                elif direction == "sell" and price_current < current['vwap']: 
                    direction = "flat"; veto_reason = "Below VWAP (Not Premium)"
                    
            # 3. فیلترهای زمان بحران
            else:
                if direction == "buy" and ml_prob_up < 0.40: direction = "flat"; veto_reason = "Crisis ML Veto"
                elif direction == "sell" and ml_prob_up > 0.60: direction = "flat"; veto_reason = "Crisis ML Veto"

        # 4. فیلتر همبستگی پیرسون (Pearson Correlation)
        if direction != "flat" and not corr_matrix.empty:
            for active_sym in active_positions:
                if symbol in corr_matrix.columns and active_sym in corr_matrix.columns:
                    corr_val = corr_matrix.loc[symbol, active_sym]
                    if abs(corr_val) > 0.85: # اگر همبستگی بالای 85 درصد بود، وتو کن
                        direction = "flat"
                        veto_reason = f"High Correlation ({corr_val:.2f}) with {active_sym}"
                        break
        
        # اگر از تمام فیلترها جان سالم به در برد، به لیست دارایی‌های فعال اضافه‌اش کن
        if direction != "flat":
            active_positions.append(symbol)

        # ======================================================================
        # سیستم تارگت‌های هوشمند و مگنت حجمی (POC)
        # ======================================================================
        dist_to_high = abs(ob_data["dealing_high"] - price_current) * 10000
        dist_to_low = abs(price_current - ob_data["dealing_low"]) * 10000
        dist_to_poc = abs(price_current - poc_current) * 10000
        
        if direction != "flat":
            if is_crisis:
                tp1_pips = round(sl_pips * 1.5, 1) 
                tp2_pips = round(sl_pips * 5.0, 1) 
            else:
                tp1_pips = round(max(dist_to_poc, sl_pips * 1.5), 1) 
                tp2_pips = round(max(dist_to_high if direction == "buy" else dist_to_low, tp1_pips * 2), 1)
        else:
            tp1_pips = round(atr_pips * 2.0, 1)
            tp2_pips = round(max(dist_to_high, dist_to_low, dist_to_poc), 1)

        portfolio_results[symbol] = {
            "regime": market_regime,
            "direction": direction,
            "target_tp1_pips": round(max(tp1_pips, 10.0), 1),
            "target_tp2_pips": round(max(tp2_pips, 20.0), 1),
            "target_sl_pips": round(max(sl_pips, 5.0), 1),
            "risk_percent": active_risk,
            "is_crisis": is_crisis,
            "veto": veto_reason
        }

    # ======================================================================
    # 8. MASTER JSON EXPORT & TELEGRAM REPORTING
    # ======================================================================
    output_data = {
        "last_update": str(datetime.datetime.utcnow()) + " UTC",
        "global_strategy": "Titan_Engine_V11",
        "session_active": session_active,
        "assets": portfolio_results
    }
    
    try:
        with open("sentinel_config.json", "w") as f:
            json.dump(output_data, f, indent=4)
        print("[SUCCESS] sentinel_config.json saved.")
    except Exception as e: 
        print(f"[ERROR] Save failed: {e}")
        
    msg = f"<b>🌍 Titan Engine V11 Live Feed</b>\n"
    msg += f"⏱ {datetime.datetime.utcnow().strftime('%H:%M UTC')}\n"
    msg += f"🏢 Session: {'🟢 Active (LDN/NY)' if session_active else '🔴 Inactive (Asian/Dead)'}\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
    
    for sym, data in portfolio_results.items():
        clean_sym = sym.replace("=X", "") 
        mode_icon = "🚨 CRISIS" if data.get('is_crisis', False) else "🛡 SMC+FVG"
        
        if data['direction'] != "flat":
            msg += f"🟢 <b>{clean_sym}</b>: <b>{data['direction'].upper()}</b> [{mode_icon}]\n"
            msg += f"   🎯 TP1: {data['target_tp1_pips']} | TP2: {data['target_tp2_pips']}\n"
            msg += f"   🛑 SL: {data['target_sl_pips']} | ⚖️ Risk: {data['risk_percent']}%\n"
        else:
            reason = data['veto'] if data['veto'] else "Monitoring OB & FVG"
            msg += f"⚪️ <b>{clean_sym}</b>: FLAT [{mode_icon}]\n"
            msg += f"   📊 Proj. Move: {data['target_tp2_pips']} pips | ⏳ <i>{reason}</i>\n"
            
        msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
            
    send_telegram_alert(msg)
    print("\n[SUCCESS] Titan Engine V11 Execution Completed.")

if __name__ == "__main__":
    generate_god_mode_strategy()
