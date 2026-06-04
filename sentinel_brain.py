import os
import json
import datetime
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# دانلود دیتابیس پردازش زبان طبیعی
nltk.download('vader_lexicon', quiet=True)

# ======================================================================
# 1. INSTITUTIONAL CONFIGURATION (تنظیمات سطح سازمان)
# ======================================================================
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE" 
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"     

# سبد دارایی‌ها
SYMBOLS = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "USDJPY=X"]

# ریسک پایه برای شرایط عادی بازار
NORMAL_RISK_PERCENT = 2.0 
# ریسک کاهش‌یافته برای زمان اخبار و بحران (با تارگت‌های نجومی)
CRISIS_RISK_PERCENT = 1.0 

def send_telegram_alert(message):
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE": return 
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try: requests.post(url, data=payload, timeout=5)
    except: pass

# ======================================================================
# 2. QUANTITATIVE INDICATORS & LIQUIDITY HUNTER (توابع ریاضی و شکارچی نقدینگی)
# ======================================================================
def calculate_vwap(df):
    """ محاسبه VWAP (میانگین قیمت وزنی حجم) برای ردگیری پول هوشمند """
    q = df['Volume'].replace(0, 1) 
    p = (df['High'] + df['Low'] + df['Close']) / 3
    return (p * q).cumsum() / q.cumsum()

def calculate_hurst(price_series, max_lag=20):
    """ تشخیص رژیم بازار (خنثی یا رونددار) """
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
    """ محاسبه فشردگی بازار برای تشخیص انفجارهای قیمتی """
    std = df['Close'].rolling(window=window).std()
    ma = df['Close'].rolling(window=window).mean()
    upper = ma + (std * 2)
    lower = ma - (std * 2)
    return (upper - lower) / ma

def check_liquidity_sweep(df, lookback=20):
    """
    سیستم جدید شکار نقدینگی (Liquidity Sweep)
    بررسی می‌کند که آیا کندل فعلی به صورت فیک (شدو) کف یا سقف 20 کندل گذشته را زده است یا خیر.
    این یکی از قوی‌ترین تاییديه‌های پرایس اکشن سازمانی است.
    """
    if len(df) < lookback + 2: return "none"
    
    recent_high = df['High'].iloc[-lookback-2:-2].max()
    recent_low = df['Low'].iloc[-lookback-2:-2].min()
    
    current = df.iloc[-2] # کندل بسته شده قبلی
    
    # سویپ سقف (لیکوید کردن فروشنده‌ها -> سیگنال نزولی)
    if current['High'] > recent_high and current['Close'] < recent_high:
        return "bearish_sweep"
    # سویپ کف (لیکوید کردن خریدارها -> سیگنال صعودی)
    elif current['Low'] < recent_low and current['Close'] > recent_low:
        return "bullish_sweep"
        
    return "none"

# ======================================================================
# 3. CRISIS NLP NEWS ENGINE (موتور تشخیص فاندامنتال و آلفای بحران)
# ======================================================================
def analyze_crisis_news():
    """
    استفاده از الگوریتم VADER برای درک لحن اخبار اقتصادی.
    در این نسخه، اگر بحرانی رخ دهد، ربات خاموش نمی‌شود، بلکه نوع بحران را شناسایی کرده 
    و سیگنال Crisis Mode صادر می‌کند تا استراتژی‌های Breakout فعال شوند.
    """
    print("[INFO] Initializing Crisis-Aware NLTK VADER Engine...")
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
        "JPY": ["BOJ", "YEN", "JAPAN", "TOKYO", "UEDA"]
    }
    
    # طبقه‌بندی هوشمند انواع بحران برای فعال‌سازی آلفای بحران
    crisis_categories = {
        "WAR_GEOPOLITIC": ['WAR', 'MISSILE', 'ATTACK', 'MILITARY', 'CEASEFIRE', 'INVASION', 'IRAN', 'ISRAEL', 'RUSSIA', 'UKRAINE', 'HEZBOLLAH', 'LEBANON'],
        "ECONOMIC_CRASH": ['CRASH', 'BLACK SWAN', 'EMERGENCY RATE CUT', 'COLLAPSE', 'BANKRUPTCY', 'LIQUIDITY CRISIS']
    }
    
    # ذخیره وضعیت بحران برای هر ارز
    crisis_status = {currency: {"active": False, "type": "none"} for currency in country_keywords.keys()}
    sentiment_scores = {currency: [] for currency in country_keywords.keys()}
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=7)
            if response.status_code != 200: continue
            root = ET.fromstring(response.content)
            
            for item in root.findall('.//item')[:40]: 
                title = (item.find('title').text or "")
                desc = (item.find('description').text or "")
                
                full_text_nlp = f"{title}. {desc}"
                full_text_upper = full_text_nlp.upper()
                
                # امتیازدهی خالص به لحن خبر
                compound_score = sia.polarity_scores(full_text_nlp)['compound']
                
                for currency, keywords in country_keywords.items():
                    if any(kw in full_text_upper for kw in keywords):
                        sentiment_scores[currency].append(compound_score)
                        
                        # جستجو برای کشف بحران‌های پول‌ساز!
                        for c_type, c_words in crisis_categories.items():
                            if any(cw in full_text_upper for cw in c_words):
                                crisis_status[currency]["active"] = True
                                crisis_status[currency]["type"] = c_type
                                print(f"[CRISIS OPPORTUNITY DETECTED] {c_type} environment for {currency}")
        except Exception as e:
            print(f"[ERROR] RSS Feed timeout: {e}")

    # ساخت ماتریس نهایی فاندامنتال
    final_matrix = {}
    for curr in country_keywords.keys():
        avg_score = np.mean(sentiment_scores[curr]) if len(sentiment_scores[curr]) > 0 else 0.0
        final_matrix[curr] = {
            "sentiment": round(avg_score, 3),
            "crisis_mode": crisis_status[curr]["active"],
            "crisis_type": crisis_status[curr]["type"]
        }
    
    print(f"[SUCCESS] Crisis NLP Matrix Built Successfully.")
    return final_matrix

# ======================================================================
# 4. QUANTITATIVE ML ENGINE (ماشین لرنینگ با ویژگی‌های حجمی)
# ======================================================================
def run_ml_prediction_for_asset(df):
    """ آموزش لحظه‌ای هوش مصنوعی روی ترکیب قیمت و حجم (VWAP) """
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
    
    if len(X) < 100: return 0.5 
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X.iloc[:-1], y.iloc[:-1])
    return model.predict_proba(X.iloc[[-1]])[0][1]

# ======================================================================
# 5. GOD MODE: 3D MULTI-TIMEFRAME ENGINE (موتور ۳ بُعدی و حلقه اصلی)
# ======================================================================
def generate_god_mode_strategy():
    print(f"\n[START] Sentinel God Mode (V7 Crisis Alpha) - {datetime.datetime.now()}")
    
    news_matrix = analyze_crisis_news()
    portfolio_results = {}
    
    for symbol in SYMBOLS:
        print(f"\n[INFO] ➜ Deep Scanning: {symbol}")
        
        df_m15 = yf.download(symbol, period="10d", interval="15m", progress=False)
        df_h1 = yf.download(symbol, period="20d", interval="1h", progress=False)
        
        if df_m15.empty or df_h1.empty: continue
            
        df_m15.columns = [col[0] if isinstance(col, tuple) else col for col in df_m15.columns]
        df_h1.columns = [col[0] if isinstance(col, tuple) else col for col in df_h1.columns]
        
        df_h4 = df_h1.resample('4h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        df_h4['ema_50'] = df_h4['Close'].ewm(span=50, adjust=False).mean()
        macro_trend = "buy" if df_h4['Close'].iloc[-1] > df_h4['ema_50'].iloc[-1] else "sell"
        
        df_h1['ema_50'] = df_h1['Close'].ewm(span=50, adjust=False).mean()
        micro_trend = "buy" if df_h1['Close'].iloc[-1] > df_h1['ema_50'].iloc[-1] else "sell"
        
        mtf_sync = (macro_trend == micro_trend)
        target_trend = macro_trend if mtf_sync else "flat"
        
        df_m15['vwap'] = calculate_vwap(df_m15)
        df_m15['ema_200'] = df_m15['Close'].ewm(span=200, adjust=False).mean()
        df_m15['rsi'] = calculate_rsi(df_m15['Close'])
        
        tr = pd.concat([df_m15['High'] - df_m15['Low'], 
                        np.abs(df_m15['High'] - df_m15['Close'].shift()), 
                        np.abs(df_m15['Low'] - df_m15['Close'].shift())], axis=1).max(axis=1)
        df_m15['atr'] = tr.rolling(14).mean()

        current = df_m15.iloc[-2] 
        prev = df_m15.iloc[-3]    

        hurst = calculate_hurst(df_m15['Close'].tail(100).values)
        market_regime = "trending" if hurst > 0.52 else "range"
        
        # اجرای شکارچی نقدینگی (آیا بانک‌ها استاپ زده‌اند؟)
        liquidity_sweep = check_liquidity_sweep(df_m15)

# چ) تاییدیه اسمارت مانی (VWAP)
        vwap_current = df_m15['vwap'].iloc[-2]
        close_current = current['Close']
        
        vwap_discount = (close_current <= vwap_current * 1.0005)
        vwap_premium = (close_current >= vwap_current * 0.9995)
        
        # ح) اجرای ماشین لرنینگ اختصاصی
        ml_prob_up = run_ml_prediction_for_asset(df_m15)
        
        direction = "flat"
        atr_pips = (current['atr'] * 10000)

        # استخراج دیتای فاندامنتال اختصاصی
        currency_code = symbol[:3]
        base_news = news_matrix.get(currency_code, {"sentiment": 0.0, "crisis_mode": False, "crisis_type": "none"})
        usd_news = news_matrix.get("USD", {"sentiment": 0.0, "crisis_mode": False, "crisis_type": "none"})
        
        # بررسی وضعیت بحران (Crisis Status)
        is_crisis = base_news["crisis_mode"] or usd_news["crisis_mode"]
        active_risk = CRISIS_RISK_PERCENT if is_crisis else NORMAL_RISK_PERCENT

        # ======================================================================
        # 6. GOD MODE DECISION MATRIX (ماتریس تصمیم‌گیری بحران‌محور)
        # ======================================================================
        if is_crisis:
            # 🔴 استراتژی آلفای بحران (شکار نوسانات شدید جنگ و اخبار)
            # در بحران، بانک‌ها ابتدا استاپ‌لاس تریدرهای خرد را می‌زنند (سویپ نقدینگی)
            if liquidity_sweep == "bullish_sweep" and target_trend != "sell":
                direction = "buy"
            elif liquidity_sweep == "bearish_sweep" and target_trend != "buy":
                direction = "sell"
            # اگر استاپ هانت نشد، اما روند 4 ساعته قدرتمند و پرشتاب است
            elif target_trend == "buy" and current['Close'] > current['ema_200']:
                direction = "buy"
            elif target_trend == "sell" and current['Close'] < current['ema_200']:
                direction = "sell"
        else:
            # 🟢 استراتژی روزهای عادی بازار (پرایس اکشن نهادی)
            if target_trend != "flat" and market_regime == "trending":
                bullish_fvg = prev['Low'] > df_m15.iloc[-4]['High']
                bearish_fvg = prev['High'] < df_m15.iloc[-4]['Low']
                
                # ترکیب روند کلان + VWAP + گپ نقدینگی
                if target_trend == "buy" and current['Close'] > current['ema_200'] and bullish_fvg and vwap_discount:
                    direction = "buy"
                elif target_trend == "sell" and current['Close'] < current['ema_200'] and bearish_fvg and vwap_premium:
                    direction = "sell"
                    
            elif target_trend == "flat" and market_regime == "range":
                bb_width = calculate_bollinger_width(df_m15).iloc[-2]
                if bb_width > 0.002: 
                    bb_std = df_m15['Close'].rolling(20).std()
                    bb_lower = df_m15['Close'].rolling(20).mean() - (bb_std * 2)
                    bb_upper = df_m15['Close'].rolling(20).mean() + (bb_std * 2)
                    
                    if current['Close'] <= bb_lower.iloc[-2] and current['rsi'] < 35: direction = "buy"
                    elif current['Close'] >= bb_upper.iloc[-2] and current['rsi'] > 65: direction = "sell"

        # ======================================================================
        # 7. DYNAMIC VETO FILTERS (فیلترهای هوشمند تطبیقی)
        # ======================================================================
        veto_reason = ""
        
        if direction != "flat":
            if not is_crisis:
                # در شرایط عادی، فیلترها سخت‌گیرانه هستند تا از معاملات ضعیف جلوگیری شود
                if direction == "buy" and ml_prob_up < 0.55: 
                    veto_reason = f"ML Vol-Prob < 55% Up ({ml_prob_up:.1%})"
                    direction = "flat"
                elif direction == "sell" and ml_prob_up > 0.45:
                    veto_reason = f"ML Vol-Prob < 55% Down ({(1-ml_prob_up):.1%})"
                    direction = "flat"
                    
                elif direction == "buy" and (base_news["sentiment"] < -0.3 and usd_news["sentiment"] > 0.3):
                    veto_reason = "NLP Mismatch (Bearish News)"
                    direction = "flat"
                elif direction == "sell" and (base_news["sentiment"] > 0.3 and usd_news["sentiment"] < -0.3):
                    veto_reason = "NLP Mismatch (Bullish News)"
                    direction = "flat"
            else:
                # در شرایط بحران، فقط اگر هوش مصنوعی صد در صد مخالف بود معامله را وتو می‌کنیم
                if direction == "buy" and ml_prob_up < 0.40:
                    veto_reason = "Crisis ML Veto (Extremely Bearish Setup)"
                    direction = "flat"
                elif direction == "sell" and ml_prob_up > 0.60:
                    veto_reason = "Crisis ML Veto (Extremely Bullish Setup)"
                    direction = "flat"

        # ======================================================================
        # 8. CRISIS-ADJUSTED TARGETS (تارگت‌های منعطف با بحران)
        # ======================================================================
        if is_crisis:
            # در زمان بحران: استاپ گشادتر (برای فرار از شدو)، تارگت فضایی برای شکار کل حرکت
            sl_pips = round(atr_pips * 2.5, 1)
            tp1_pips = round(atr_pips * 2.5, 1) 
            tp2_pips = round(atr_pips * 6.0, 1) # شکار 6 برابر ریسک!
        else:
            # روزهای عادی
            sl_pips = round(atr_pips * 1.5, 1)
            tp1_pips = round(atr_pips * 1.5, 1) 
            tp2_pips = round(atr_pips * 4.0, 1) if market_regime == "trending" else round(atr_pips * 2.5, 1)
            
        # ذخیره در دیکشنری نهایی سبد
        portfolio_results[symbol] = {
            "regime": market_regime,
            "direction": direction,
            "target_tp1_pips": max(tp1_pips, 10.0),
            "target_tp2_pips": max(tp2_pips, 20.0),
            "target_sl_pips": max(sl_pips, 12.0),
            "risk_percent": active_risk,
            "is_crisis": is_crisis,
            "veto": veto_reason
        }

# ======================================================================
    # 9. MASTER JSON OUTPUT & TELEGRAM REPORTING (خروجی و گزارش‌گیری کلان)
    # ======================================================================
    # اکنون از حلقه for (پردازش 5 ارز) خارج شده‌ایم
    
    # ساختار نهایی فایل JSON برای خوانش هم‌زمان در متاتریدر 5
    output_data = {
        "last_update": str(datetime.datetime.now()),
        "assets": portfolio_results
    }
    
    # ذخیره فایل در مخزن گیت‌هاب
    with open("sentinel_config.json", "w") as f:
        json.dump(output_data, f, indent=4)
        
    # ساخت گزارش شگفت‌انگیز و حرفه‌ای برای تلگرام
    msg = f"<b>🏛 Sentinel God Mode (V7 Crisis Alpha)</b>\n"
    msg += f"⏱ {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
    
    for sym, data in portfolio_results.items():
        clean_sym = sym.replace("=X", "") # پاک کردن =X برای زیبایی
        
        # برچسب‌گذاری گرافیکی برای اینکه بفهمیم ربات در چه حالتی است
        mode_icon = "🚨 CRISIS MODE" if data.get('is_crisis', False) else "🛡 NORMAL MODE"
        
        if data['direction'] != "flat":
            # ارزهایی که سیگنال ورود دارند
            msg += f"🟢 <b>{clean_sym}</b>: <b>{data['direction'].upper()}</b> [{mode_icon}]\n"
            msg += f"   🎯 TP1: {data['target_tp1_pips']} | TP2: {data['target_tp2_pips']}\n"
            msg += f"   🛑 SL: {data['target_sl_pips']} | ⚖️ Risk: {data['risk_percent']}%\n"
        else:
            # ارزهایی که منتظر فرصت هستند یا فیلتر شده‌اند
            reason = data['veto'] if data['veto'] else "Awaiting Setup"
            msg += f"⚪️ <b>{clean_sym}</b>: FLAT [{mode_icon}]\n"
            msg += f"   ⏳ <i>{reason}</i>\n"
            
        msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
            
    send_telegram_alert(msg)
    print("\n[SUCCESS] V7 Crisis Alpha Execution Completed. JSON generated and Telegram sent.")

# ======================================================================
# اجرای اصلی برنامه
# ======================================================================
if __name__ == "__main__":
    generate_god_mode_strategy()
