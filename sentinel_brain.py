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

# دانلود دیتابیس کلمات پردازش زبان طبیعی (فقط برای اجرای اول نیاز است)
nltk.download('vader_lexicon', quiet=True)

# ======================================================================
# 1. INSTITUTIONAL CONFIGURATION & TIME CORES
# ======================================================================

# توکن و آیدی تلگرام برای ارسال گزارش‌های زنده
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE" 
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"     

# سبد جامع دارایی‌های جهانی (۱۰ جفت‌ارز قدرتمند)
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

# ریسک پایه برای زمانی که ماشین لرنینگ در دسترس نیست یا سیگنال ضعیف است
DEFAULT_RISK_PERCENT = 1.0 

# ساعات طلایی نقدینگی (سشن لندن و نیویورک به وقت UTC)
# خارج از این ساعات، بازار پر از اسپرد و نویز است
TRADING_HOURS = {
    "london_start": 7,  # شروع سشن لندن
    "new_york_end": 21  # پایان سشن نیویورک
}

def is_market_liquid_now():
    """
    بررسی می‌کند که آیا در حال حاضر بازار در زمان اوج حجم و نقدینگی قرار دارد یا خیر.
    این تابع از ورود به معاملات در سشن‌های مرده (مثل اواسط سشن آسیا) جلوگیری می‌کند.
    """
    current_hour = datetime.datetime.utcnow().hour
    current_day = datetime.datetime.utcnow().weekday()
    
    # روزهای شنبه و یکشنبه (5 و 6 در پایتون) بازار جهانی تعطیل است
    if current_day >= 5: 
        return False
        
    # فقط در بازه زمانی تعیین شده اجازه معامله صادر می‌شود
    if TRADING_HOURS["london_start"] <= current_hour <= TRADING_HOURS["new_york_end"]:
        return True
    else:
        return False

def send_telegram_alert(message):
    """
    ارسال پیام‌های فرمت‌بندی شده (HTML) به ربات تلگرام مدیر سیستم.
    """
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
# 2. ADVANCED RISK MANAGEMENT & CORRELATION
# ======================================================================

def calculate_kelly_criterion(win_probability, tp_pips, sl_pips):
    """
    محاسبه حجم بهینه معامله با استفاده از فرمول Kelly Criterion.
    این فرمول بر اساس احتمال برد (که از هوش مصنوعی می‌گیریم) و نسبت ریوارد به ریسک،
    مشخص می‌کند دقیقاً چه درصدی از سرمایه باید درگیر شود.
    """
    if sl_pips == 0 or win_probability < 0.50: 
        return 0.0
    
    # محاسبه نسبت سود به ضرر (Reward to Risk Ratio)
    b = tp_pips / sl_pips
    
    # احتمال برد (p) و احتمال باخت (q)
    p = win_probability
    q = 1.0 - p
    
    # فرمول اصلی کلی: f* = p - (q / b)
    kelly_fraction = p - (q / b)
    
    # برای امنیت سرمایه، از نصف کلی (Half-Kelly) استفاده می‌کنیم
    half_kelly_percent = (kelly_fraction / 2.0) * 100
    
    # محدود کردن ریسک برای جلوگیری از دراوداون‌های خطرناک (بین 0.5% تا 3.0%)
    final_risk = max(min(half_kelly_percent, 3.0), 0.5)
    
    return round(final_risk, 2)

def calculate_portfolio_correlation(symbols_list, lookback_days=10):
    """
    محاسبه ماتریس همبستگی پیرسون (Pearson Correlation Matrix).
    این تابع بررسی می‌کند که آیا ارزها در روزهای اخیر حرکات مشابهی داشته‌اند یا خیر.
    اگر همبستگی دو ارز بالای 85% باشد، ربات فقط روی یکی از آن‌ها معامله می‌کند تا ریسک دوبل نشود.
    """
    print("[INFO] Computing Live Pearson Correlation Matrix...")
    close_prices = {}
    
    for sym in symbols_list:
        try:
            df = yf.download(sym, period=f"{lookback_days}d", interval="1h", progress=False)
            if not df.empty:
                # پاک‌سازی ساختار داده‌های یاهو فایننس
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                close_prices[sym] = df['Close']
        except Exception as e: 
            print(f"[WARNING] Could not fetch correlation data for {sym}: {e}")
            pass
            
    if len(close_prices) < 2: 
        return pd.DataFrame()
        
    correlation_matrix = pd.DataFrame(close_prices).corr(method='pearson')
    print("[SUCCESS] Correlation Matrix Calculated.")
    
    return correlation_matrix

# ======================================================================
# 3. QUANTITATIVE INDICATORS, FVG & VSA (اندیکاتورها، گپ‌ها و حجم)
# ======================================================================

def calculate_vwap(df):
    """
    محاسبه میانگین قیمت وزنی حجم (VWAP).
    این خط به ما نشان می‌دهد که از دیدگاه نهادهای مالی بزرگ و الگوریتم‌های بانکی،
    ارزش واقعی و منصفانه این دارایی در حال حاضر چقدر است.
    """
    # جایگزینی صفرهای احتمالی در حجم با 1 برای جلوگیری از خطای تقسیم بر صفر
    q = df['Volume'].replace(0, 1) 
    p = (df['High'] + df['Low'] + df['Close']) / 3
    
    vwap = (p * q).cumsum() / q.cumsum()
    return vwap

def calculate_poc(df, lookback=100):
    """
    محاسبه نقطه کنترل (Point of Control - POC) از ولوم پروفایل.
    این تابع بررسی می‌کند که در 100 کندل گذشته، بیشترین حجم پول در چه قیمت دقیقی معامله شده است.
    این قیمت مانند یک آهنربای قدرتمند برای تارگت‌ها عمل می‌کند.
    """
    if len(df) < lookback: 
        return df['Close'].iloc[-1]
        
    recent = df.iloc[-lookback:].copy()
    
    if recent['Volume'].sum() == 0: 
        return recent['Close'].mean()
        
    # گروه‌بندی قیمت‌ها تا 4 رقم اعشار برای پیدا کردن دقیق‌ترین سطح درگیری پول
    recent['Price_Bin'] = recent['Close'].round(4)
    poc_price = recent.groupby('Price_Bin')['Volume'].sum().idxmax()
    
    return poc_price

def calculate_hurst(price_series, max_lag=20):
    """
    تشخیص رژیم بازار با فرمول Hurst Exponent.
    اگر عدد بالاتر از 0.52 باشد، بازار رونددار (Trending) است.
    اگر کمتر باشد، بازار خنثی و پر از نویز (Range) است.
    """
    if len(price_series) < max_lag: 
        return 0.5
        
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    
    hurst_value = poly[0] * 2.0
    return hurst_value

def calculate_rsi(data, periods=14):
    """ محاسبه اندیکاتور قدرت نسبی (RSI) برای استفاده در ماشین لرنینگ. """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_width(df, window=20):
    """
    محاسبه فشردگی باندهای بولینگر (Bollinger Band Width).
    کاهش شدید این عدد نشان‌دهنده آرامش قبل از طوفان و احتمال یک حرکت شارپ است.
    """
    std = df['Close'].rolling(window=window).std()
    ma = df['Close'].rolling(window=window).mean()
    upper = ma + (std * 2)
    lower = ma - (std * 2)
    
    bb_width = (upper - lower) / ma
    return bb_width

def detect_vsa_anomaly(df):
    """
    تحلیل حجم و اسپرد (Volume Spread Analysis - VSA).
    بررسی می‌کند که آیا کندل فعلی دارای ورود حجم غیرعادی و سنگین (پول هوشمند) هست یا خیر.
    """
    if len(df) < 20: 
        return False
        
    vol_sma = df['Volume'].rolling(20).mean().iloc[-2]
    current_vol = df['Volume'].iloc[-2]
    
    # اگر حجم کندل فعلی 50 درصد بیشتر از میانگین 20 کندل اخیر باشد، تایید می‌شود
    if current_vol > (vol_sma * 1.5):
        return True
    return False

def check_liquidity_sweep(df, lookback=20):
    """
    سیستم شکار نقدینگی (Stop Hunt Detection).
    تشخیص می‌دهد که آیا قیمت با یک شدو (Wick) استاپ‌لاس‌های سقف یا کف را زده و برگشته است؟
    """
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
    """
    کشف گپ‌های ارزش منصفانه (Fair Value Gaps - FVG) در پرایس اکشن سازمانی.
    به دنبال عدم تعادل‌های قیمتی (Imbalance) می‌گردد که نشان‌دهنده قدرت مطلق خریداران یا فروشندگان است.
    """
    fvgs = {"bullish": [], "bearish": []}
    if len(df) < lookback: 
        return fvgs
        
    recent = df.iloc[-lookback:-1]
    
    for i in range(1, len(recent)-1):
        # بررسی FVG صعودی (فاصله بین سقف کندل اول و کف کندل سوم)
        if recent['Low'].iloc[i+1] > recent['High'].iloc[i-1]:
            if recent['Close'].iloc[i] > recent['Open'].iloc[i]:
                fvgs["bullish"].append((recent['High'].iloc[i-1], recent['Low'].iloc[i+1]))
                
        # بررسی FVG نزولی (فاصله بین کف کندل اول و سقف کندل سوم)
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
    این تابع فقط اوردر بلاک‌هایی را معتبر می‌داند که باعث شکست ساختار (BOS) شده باشند
    و در مناطق صحیح (تخفیف برای خرید، گران برای فروش) قرار گرفته باشند.
    """
    if len(df) < lookback: 
        return None
        
    recent = df.iloc[-lookback:-1].copy()
    
    # 1. تعیین محدوده معاملاتی زنده و نقطه تعادل (Equilibrium)
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
    
    # 2. جستجو برای اوردر بلاک صعودی (در منطقه ارزان‌فروشی / Discount Zone)
    # به دنبال آخرین کندل نزولی قبل از یک رالی قدرتمند می‌گردیم که سقف‌ها را شکسته باشد
    for i in range(len(recent) - 5, 2, -1):
        cond_down_candle = recent['Close'].iloc[i] < recent['Open'].iloc[i]
        cond_strong_move = (recent['Close'].iloc[i+1] > recent['High'].iloc[i] and 
                            recent['Close'].iloc[i+2] > recent['Close'].iloc[i+1])
        
        if cond_down_candle and cond_strong_move:
            # بررسی اینکه آیا اوردر بلاک واقعاً در منطقه ارزان قرار دارد؟
            if recent['High'].iloc[i] < equilibrium:
                order_blocks["bullish_ob_top"] = recent['High'].iloc[i]
                order_blocks["bullish_ob_bottom"] = recent['Low'].iloc[i]
                order_blocks["bullish_valid"] = True
                break
                
    # 3. جستجو برای اوردر بلاک نزولی (در منطقه گران‌فروشی / Premium Zone)
    # به دنبال آخرین کندل صعودی قبل از یک ریزش قدرتمند می‌گردیم که کف‌ها را شکسته باشد
    for i in range(len(recent) - 5, 2, -1):
        cond_up_candle = recent['Close'].iloc[i] > recent['Open'].iloc[i]
        cond_sharp_drop = (recent['Close'].iloc[i+1] < recent['Low'].iloc[i] and 
                           recent['Close'].iloc[i+2] < recent['Close'].iloc[i+1])
        
        if cond_up_candle and cond_sharp_drop:
            # بررسی اینکه آیا اوردر بلاک واقعاً در منطقه گران قرار دارد؟
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
    این مدل با استفاده از 4 متغیر کلیدی (RSI، فشردگی بولینگر، فاصله از VWAP و بازدهی)
    احتمال صعودی بودن کندل بعدی را با دقت بالا محاسبه می‌کند.
    """
    df_ml = df.copy()
    
    # ساخت ویژگی‌ها (Feature Engineering)
    df_ml['rsi'] = calculate_rsi(df_ml['Close'])
    df_ml['bb_width'] = calculate_bollinger_width(df_ml)
    vwap_line = calculate_vwap(df_ml)
    df_ml['vwap_dist'] = df_ml['Close'] - vwap_line
    df_ml['returns'] = df_ml['Close'].pct_change()
    
    # تعریف هدف (Target): آیا کندل بعدی صعودی خواهد بود؟ (1=بله، 0=خیر)
    df_ml['target'] = np.where(df_ml['Close'].shift(-1) > df_ml['Close'], 1, 0)
    
    # حذف ردیف‌هایی که دیتای ناقص دارند (به دلیل شیفت دادن)
    df_ml.dropna(inplace=True)
    
    X = df_ml[['rsi', 'bb_width', 'vwap_dist', 'returns']]
    y = df_ml['target']
    
    # اگر دیتای کافی برای آموزش مدل وجود نداشته باشد، عدد خنثی (50%) برمی‌گرداند
    if len(X) < 100: 
        return 0.5
    
    # پیکربندی مدل XGBoost برای جلوگیری از نویز و بیش‌برازش (Overfitting)
    model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.05, 
        random_state=42, 
        eval_metric='logloss'
    )
    
    # آموزش مدل روی تمام داده‌های گذشته و پیش‌بینی وضعیت کندل فعلی
    model.fit(X.iloc[:-1], y.iloc[:-1])
    
    # برگرداندن احتمال صعودی بودن قیمت (عددی بین 0.0 تا 1.0)
    prediction_probability = model.predict_proba(X.iloc[[-1]])[0][1]
    
    return prediction_probability


# ======================================================================
# 7. QUANTUM ENGINE: MAIN STRATEGY & EXECUTION (موتور اصلی و اجرای استراتژی)
# ======================================================================

def generate_god_mode_strategy():
    """
    موتور اصلی سیستم (Quantum Engine V12).
    این تابع تمام ماژول‌های قبلی (NLP, SMC, FVG, VSA, XGBoost) را با هم ترکیب کرده
    و زمان دقیق انقضا و قیمت ورود را برای متاتریدر صادر می‌کند.
    """
    # ثبت زمان دقیق اجرا (به وقت جهانی) برای محاسبه تاریخ انقضای سیگنال
    run_time_utc = datetime.datetime.utcnow()
    print(f"\n[START] Quantum Engine V12 - {run_time_utc.strftime('%Y-%m-%d %H:%M UTC')}")
    
    # 1. بررسی وضعیت نقدینگی و سشن‌های معاملاتی
    session_active = is_market_liquid_now()
    print(f"[INFO] Session Active: {session_active}")

    # 2. استخراج داده‌های کلان (فاندامنتال و ماتریس همبستگی)
    news_matrix = analyze_crisis_news()
    corr_matrix = calculate_portfolio_correlation(SYMBOLS)
    
    portfolio_results = {}
    active_positions = [] # لیستی برای ثبت دارایی‌هایی که سیگنال معتبر دارند (جهت مهار ریسک)
    
    # حلقه اصلی: بررسی تک تک 10 دارایی جهانی
    for symbol in SYMBOLS:
        print(f"[INFO] Deep Scanning: {symbol}")
        
        # دریافت داده‌های 15 دقیقه‌ای و 1 ساعته
        df_m15 = yf.download(symbol, period="15d", interval="15m", progress=False)
        df_h1 = yf.download(symbol, period="30d", interval="1h", progress=False)
        
        if df_m15.empty or df_h1.empty: 
            continue
            
        # پاک‌سازی ساختار ستون‌های یاهو فایننس
        df_m15.columns = [col[0] if isinstance(col, tuple) else col for col in df_m15.columns]
        df_h1.columns = [col[0] if isinstance(col, tuple) else col for col in df_h1.columns]
        
        # ساخت تایم فریم 4 ساعته با روش Resampling
        df_h4 = df_h1.resample('4h').agg({
            'Open': 'first', 
            'High': 'max', 
            'Low': 'min', 
            'Close': 'last', 
            'Volume': 'sum'
        }).dropna()
        
        # تحلیل روند کلان (Macro Trend) روی 4 ساعته
        df_h4['ema_50'] = df_h4['Close'].ewm(span=50, adjust=False).mean()
        macro_trend = "buy" if df_h4['Close'].iloc[-1] > df_h4['ema_50'].iloc[-1] else "sell"
        
        # تحلیل روند خرد (Micro Trend) روی 1 ساعته
        df_h1['ema_50'] = df_h1['Close'].ewm(span=50, adjust=False).mean()
        micro_trend = "buy" if df_h1['Close'].iloc[-1] > df_h1['ema_50'].iloc[-1] else "sell"
        
        # هم‌سویی روندها
        target_trend = macro_trend if (macro_trend == micro_trend) else "flat"
        
        # محاسبات کوانت و اسمارت مانی
        df_m15['vwap'] = calculate_vwap(df_m15)
        df_m15['bb_width'] = calculate_bollinger_width(df_m15)
        poc_current = calculate_poc(df_m15, lookback=100)
        
        # محاسبه ATR برای حد ضررهای داینامیک
        tr = pd.concat([
            df_m15['High'] - df_m15['Low'], 
            np.abs(df_m15['High'] - df_m15['Close'].shift()), 
            np.abs(df_m15['Low'] - df_m15['Close'].shift())
        ], axis=1).max(axis=1)
        df_m15['atr'] = tr.rolling(14).mean()

        current = df_m15.iloc[-2] # کندل کاملاً بسته شده قبلی
        
        # محاسبه رژیم بازار
        hurst = calculate_hurst(df_m15['Close'].tail(100).values)
        market_regime = "trending" if hurst > 0.52 else "range"
        
        # استخراج داده‌های پیشرفته (نقدینگی، گپ‌ها، اوردر بلاک‌ها و حجم)
        liquidity_sweep = check_liquidity_sweep(df_m15)
        fvgs = detect_fair_value_gaps(df_m15)
        ob_data = detect_institutional_order_blocks(df_m15)
        vsa_anomaly = detect_vsa_anomaly(df_m15) 
        
        if not ob_data: 
            continue
        
        # اجرای هوش مصنوعی XGBoost برای این جفت‌ارز
        ml_prob_up = run_ml_prediction_for_asset(df_m15)
        
        direction = "flat"
        atr_pips = (current['atr'] * 10000)
        price_current = current['Close']
        bb_width_current = current['bb_width']
        
        # جداسازی ارز پایه و متقابل برای بررسی اخبار
        base_curr = symbol[:3]
        quote_curr = symbol[3:6]
        
        base_news = news_matrix.get(base_curr, {"sentiment": 0.0, "crisis_mode": False})
        quote_news = news_matrix.get(quote_curr, {"sentiment": 0.0, "crisis_mode": False})
        
        # اگر هر یک از ارزهای این جفت درگیر بحران باشند، حالت Crisis فعال می‌شود
        is_crisis = base_news["crisis_mode"] or quote_news["crisis_mode"]

        # بررسی نفوذ قیمت به داخل اوردر بلاک‌های معتبر (Mitigation)
        testing_bullish_ob = ob_data["bullish_valid"] and (ob_data["bullish_ob_bottom"] <= price_current <= ob_data["bullish_ob_top"])
        testing_bearish_ob = ob_data["bearish_valid"] and (ob_data["bearish_ob_bottom"] <= price_current <= ob_data["bearish_ob_top"])

        # تنظیم استاپ‌لاس پیش‌فرض و ثبت قیمت ورود دقیق (برای ارسال به متاتریدر)
        sl_pips = round(atr_pips * 1.5, 1)
        exact_entry_price = round(price_current, 5) 

        # ======================================================================
        # ماتریس ورود پیشرفته (Entry Matrix)
        # ======================================================================
        if is_crisis:
            # در زمان بحران، به جای استراتژی عادی، استاپ‌هانت‌ها را شکار می‌کنیم
            if liquidity_sweep == "bullish_sweep" and target_trend != "sell":
                direction = "buy"
                sl_pips = round(atr_pips * 2.5, 1) # استاپ عریض‌تر برای فرار از نوسانات جنگ
            elif liquidity_sweep == "bearish_sweep" and target_trend != "buy":
                direction = "sell"
                sl_pips = round(atr_pips * 2.5, 1)
        else:
            # در روزهای عادی: ورود مستلزم هم‌سویی روند، اوردر بلاک معتبر و گپ باز (FVG) است
            if target_trend == "buy" and testing_bullish_ob and len(fvgs["bullish"]) > 0:
                direction = "buy"
                sl_price = min(ob_data["bullish_ob_bottom"], ob_data["dealing_low"])
                sl_pips = max(abs(price_current - sl_price) * 10000 + 2.0, 10.0) 
                
            elif target_trend == "sell" and testing_bearish_ob and len(fvgs["bearish"]) > 0:
                direction = "sell"
                sl_price = max(ob_data["bearish_ob_top"], ob_data["dealing_high"])
                sl_pips = max(abs(sl_price - price_current) * 10000 + 2.0, 10.0)

        # ======================================================================
        # ماتریس فیلترها و وتو (Risk & Veto Matrix)
        # ======================================================================
        # آستانه داینامیک هوش مصنوعی بر اساس نوسان بازار
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
                    direction = "flat"
                    veto_reason = f"XGBoost Veto (<{dynamic_threshold:.2f})"
                elif direction == "sell" and ml_prob_up > (1 - dynamic_threshold): 
                    direction = "flat"
                    veto_reason = f"XGBoost Veto (>{(1-dynamic_threshold):.2f})"
                elif direction == "buy" and price_current > current['vwap']: 
                    direction = "flat"
                    veto_reason = "Above VWAP (Not Discounted)"
                elif direction == "sell" and price_current < current['vwap']: 
                    direction = "flat"
                    veto_reason = "Below VWAP (Not Premium)"
                    
            # 3. فیلترهای زمان بحران
            else:
                if direction == "buy" and ml_prob_up < 0.40: 
                    direction = "flat"
                    veto_reason = "Crisis ML Veto"
                elif direction == "sell" and ml_prob_up > 0.60: 
                    direction = "flat"
                    veto_reason = "Crisis ML Veto"

        # 4. فیلتر همبستگی پیرسون (جلوگیری از ریسک مضاعف روی ارزهای مشابه)
        if direction != "flat" and not corr_matrix.empty:
            for active_sym in active_positions:
                if symbol in corr_matrix.columns and active_sym in corr_matrix.columns:
                    corr_val = corr_matrix.loc[symbol, active_sym]
                    if abs(corr_val) > 0.85: # همبستگی بسیار بالا
                        direction = "flat"
                        veto_reason = f"Correlation Clash ({corr_val:.2f}) with {active_sym}"
                        break
        
        # ثبت دارایی در صورت تایید نهایی
        if direction != "flat": 
            active_positions.append(symbol)

        # ======================================================================
        # سیستم تارگت‌ها، مدیریت ریسک Kelly و تاریخ انقضا
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
            
            # محاسبه ریسک بر اساس فرمول Kelly
            win_prob = ml_prob_up if direction == "buy" else (1 - ml_prob_up)
            active_risk = calculate_kelly_criterion(win_prob, tp1_pips, sl_pips)
            
            # تاییدیه VSA (ورود حجم سنگین بانک‌ها) ریسک را کمی افزایش می‌دهد (اطمینان بیشتر)
            if vsa_anomaly: 
                active_risk = round(min(active_risk * 1.2, 3.0), 2)
                
        else:
            tp1_pips = round(atr_pips * 2.0, 1)
            tp2_pips = round(max(dist_to_high, dist_to_low, dist_to_poc), 1)
            active_risk = DEFAULT_RISK_PERCENT

        # ایجاد زمان انقضا برای سیگنال (اعتبار فقط تا 2 ساعت پس از صدور)
        signal_expiry_time = run_time_utc + timedelta(hours=2)
        signal_expiry_str = signal_expiry_time.strftime('%Y-%m-%d %H:%M:%S')

        # ثبت نتایج در دیکشنری نهایی برای این ارز
        portfolio_results[symbol] = {
            "regime": market_regime,
            "direction": direction,
            "entry_price": exact_entry_price,       # متغیر جدید: قیمت دقیق
            "expiration_utc": signal_expiry_str,    # متغیر جدید: زمان انقضا
            "target_tp1_pips": round(max(tp1_pips, 10.0), 1),
            "target_tp2_pips": round(max(tp2_pips, 20.0), 1),
            "target_sl_pips": round(max(sl_pips, 5.0), 1),
            "risk_percent": active_risk,            # متغیر جدید: ریسک داینامیک Kelly
            "is_crisis": is_crisis,
            "vsa_confirmed": bool(vsa_anomaly),     # متغیر جدید: تاییدیه حجم
            "veto": veto_reason
        }

    # ======================================================================
    # 8. MASTER JSON EXPORT & TELEGRAM REPORTING (خروجی و گزارش‌گیری)
    # ======================================================================
    
    # اکنون از حلقه for (بررسی ارزها) خارج شده‌ایم
    
    # ساختار نهایی فایل JSON با تمام متغیرهای سازمانی و مدیریت زمان
    output_data = {
        "last_update": run_time_utc.strftime('%Y-%m-%d %H:%M:%S'),
        "global_strategy": "Quantum_Engine_V12",
        "session_active": session_active,
        "assets": portfolio_results
    }
    
    # ذخیره‌سازی امن فایل JSON برای خوانش توسط متاتریدر 5
    try:
        with open("sentinel_config.json", "w") as f: 
            json.dump(output_data, f, indent=4)
        print("[SUCCESS] sentinel_config.json saved successfully.")
    except Exception as e: 
        print(f"[ERROR] JSON Save failed: {e}")
        
    # ساخت گزارش داشبورد فوق‌حرفه‌ای برای تلگرام
    msg = f"<b>🌍 Quantum Engine V12 Live Feed</b>\n"
    msg += f"⏱ {run_time_utc.strftime('%H:%M UTC')}\n"
    
    # نمایش وضعیت سشن معاملاتی
    if session_active:
        msg += f"🏢 Session: 🟢 Active (London/New York)\n"
    else:
        msg += f"🏢 Session: 🔴 Inactive (Asian/Dead Zone)\n"
        
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
    
    for sym, data in portfolio_results.items():
        clean_sym = sym.replace("=X", "") 
        
        # آیکون‌های وضعیت بحران و تاییدیه حجم پول هوشمند (VSA)
        mode_icon = "🚨 CRISIS" if data.get('is_crisis', False) else "🛡 SMC+FVG"
        vol_icon = "🔥 VSA" if data.get('vsa_confirmed', False) else "📊"
        
        if data['direction'] != "flat":
            # ارزهایی که سیگنال ورود قطعی دارند
            msg += f"🟢 <b>{clean_sym}</b>: <b>{data['direction'].upper()}</b> [{mode_icon}]\n"
            msg += f"   📍 Entry: {data['entry_price']} [{vol_icon}]\n"
            msg += f"   🎯 TP1: {data['target_tp1_pips']} | TP2: {data['target_tp2_pips']}\n"
            msg += f"   🛑 SL: {data['target_sl_pips']} | ⚖️ Risk: {data['risk_percent']}%\n"
            msg += f"   ⏳ Expires: {data['expiration_utc']}\n"
        else:
            # ارزهایی که در حال مانیتورینگ هستند یا با فیلترها وتو شده‌اند
            reason = data['veto'] if data['veto'] else "Monitoring OB & FVG"
            msg += f"⚪️ <b>{clean_sym}</b>: FLAT [{mode_icon}]\n"
            msg += f"   📊 Proj. Move: {data['target_tp2_pips']} pips\n"
            msg += f"   ⏳ <i>{reason}</i>\n"
            
        msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
            
    # ارسال پیام به تلگرام مدیر
    send_telegram_alert(msg)
    print("\n[SUCCESS] Quantum Engine V12 Execution Completed.")

# ======================================================================
# 9. MAIN EXECUTION BLOCK (نقطه شروع اجرای اسکریپت)
# ======================================================================
if __name__ == "__main__":
    generate_god_mode_strategy()
