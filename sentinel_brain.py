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

# مدیریت سرمایه داینامیک: حداکثر ۲ درصد ریسک در هر معامله
# متاتریدر با استفاده از این عدد، خودش لات سایز را بر اساس ۱۰۰ دلار یا ۱۰۰۰ دلار محاسبه می‌کند
RISK_PER_TRADE_PERCENT = 2.0 

def send_telegram_alert(message):
    """ ارسال گزارشات شیفت‌بندی شده به تلگرام """
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE": return 
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try: requests.post(url, data=payload, timeout=5)
    except: pass

# ======================================================================
# 2. QUANTITATIVE INDICATORS (اندیکاتورهای پیشرفته کوانت)
# ======================================================================
def calculate_vwap(df):
    """
    محاسبه VWAP (میانگین قیمت وزنی حجم)
    مهم‌ترین اندیکاتور بانک‌ها برای تشخیص ارزش واقعی دارایی
    """
    q = df['Volume']
    p = (df['High'] + df['Low'] + df['Close']) / 3
    # در صورت صفر بودن حجم (تعطیلات)، برای جلوگیری از ارور از 1 استفاده می‌کنیم
    q = q.replace(0, 1) 
    vwap = (p * q).cumsum() / q.cumsum()
    return vwap

def calculate_hurst(price_series, max_lag=20):
    """ تشخیص رژیم بازار (ساید یا رونددار) """
    if len(price_series) < max_lag: return 0.5
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def calculate_rsi(data, periods=14):
    """ محاسبه RSI کلاسیک """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_width(df, window=20):
    """
    محاسبه فشردگی باندهای بولینگر (Squeeze)
    برای تشخیص زمان‌هایی که بازار آماده یک انفجار قیمتی است
    """
    std = df['Close'].rolling(window=window).std()
    ma = df['Close'].rolling(window=window).mean()
    upper = ma + (std * 2)
    lower = ma - (std * 2)
    return (upper - lower) / ma


# ======================================================================
# 3. INSTITUTIONAL NLP NEWS ENGINE (موتور فاندامنتال سطح سازمان)
# ======================================================================
def analyze_institutional_news():
    """
    استفاده از الگوریتم VADER برای درک لحن اخبار اقتصادی،
    تشخیص تفاوت افعال منفی و مثبت، و ایجاد یک سپر محافظ (Circuit Breaker)
    برای جلوگیری از نابودی حساب در زمان رویدادهای «قوی سیاه».
    """
    print("[INFO] Initializing Institutional NLTK VADER Engine...")
    sia = SentimentIntensityAnalyzer()
    
    # فیدهای خبری سریع‌تر و معتبرتر
    urls = [
        "https://www.fxstreet.com/rss/news",
        "https://www.forexlive.com/feed/news"
    ]
    
    # نگاشت هوشمند ارزها به اقتصادهای مربوطه
    country_keywords = {
        "USD": ["FED", "FOMC", "CPI", "NFP", "POWELL", "US ", "TREASURY", "DOLLAR"],
        "EUR": ["ECB", "LAGARDE", "EUROZONE", "GERMANY", "FRANCE", "EURO "],
        "GBP": ["BOE", "BAILEY", "UK ", "BRITAIN", "BREXIT", "STERLING"],
        "AUD": ["RBA", "AUSTRALIA", "AUSSIE", "SYDNEY"],
        "CAD": ["BOC", "CANADA", "LOONIE", "OTTAWA", "OIL"],
        "JPY": ["BOJ", "YEN", "JAPAN", "TOKYO", "UEDA"]
    }
    
    circuit_breakers = {currency: False for currency in country_keywords.keys()}
    sentiment_scores = {currency: [] for currency in country_keywords.keys()}
    
    # کلمات ماشه‌ای (Trigger Words) که باعث توقف آنی تمام معاملات یک ارز می‌شوند
    critical_events = ['EMERGENCY', 'CRASH', 'INTERVENTION', 'WAR', 'SHOCK', 'COLLAPSE', 'BLACK SWAN']

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=7)
            if response.status_code != 200: continue
            root = ET.fromstring(response.content)
            
            # بررسی عمیق‌تر بازار (40 خبر آخر) برای اطمینان از پوشش تمام 5 ارز
            for item in root.findall('.//item')[:40]: 
                title = (item.find('title').text or "")
                desc = (item.find('description').text or "")
                
                # متن کامل برای هوش مصنوعی (حساس به ساختار جمله)
                full_text_nlp = f"{title}. {desc}"
                full_text_upper = full_text_nlp.upper()
                
                # امتیازدهی خالص به لحن خبر (-1 تا +1)
                sentiment_dict = sia.polarity_scores(full_text_nlp)
                compound_score = sentiment_dict['compound']
                
                for currency, keywords in country_keywords.items():
                    if any(kw in full_text_upper for kw in keywords):
                        sentiment_scores[currency].append(compound_score)
                        
                        # فعال‌سازی قطع‌کننده خودکار در صورت بروز فاجعه اقتصادی
                        if any(ce in full_text_upper for ce in critical_events):
                            circuit_breakers[currency] = True
                            print(f"[CRITICAL ALERT] Circuit Breaker triggered for {currency}: {title}")
        except Exception as e:
            print(f"[ERROR] RSS Feed timeout or parsing failed: {e}")

    # ساخت ماتریس نهایی سنتیمنت
    final_matrix = {}
    for curr in country_keywords.keys():
        avg_score = np.mean(sentiment_scores[curr]) if len(sentiment_scores[curr]) > 0 else 0.0
        final_matrix[curr] = {
            "sentiment": round(avg_score, 3),
            "blocked": circuit_breakers[curr]
        }
    
    print(f"[SUCCESS] Institutional NLP Matrix Generated.")
    return final_matrix

# ======================================================================
# 4. QUANTITATIVE ML ENGINE (ماشین لرنینگ با ویژگی‌های حجمی)
# ======================================================================
def run_ml_prediction_for_asset(df):
    """
    آموزش لحظه‌ای هوش مصنوعی روی ترکیب قیمت و حجم (VWAP)
    هوش مصنوعی در این نسخه فشردگی بازار و ارزش منصفانه بانک‌ها را یاد می‌گیرد
    """
    df_ml = df.copy()
    
    # استخراج ویژگی‌های پیشرفته (Feature Engineering)
    df_ml['rsi'] = calculate_rsi(df_ml['Close'])
    df_ml['bb_width'] = calculate_bollinger_width(df_ml) # فشردگی بازار
    
    vwap_line = calculate_vwap(df_ml)
    df_ml['vwap_dist'] = df_ml['Close'] - vwap_line # فاصله قیمت تا VWAP
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
    print(f"\n[START] Sentinel God Mode (V6) - {datetime.datetime.now()}")
    
    # 1. خواندن ذهن بازار جهانی (NLP) و استخراج سنتیمنت‌ها
    news_matrix = analyze_institutional_news()
    portfolio_results = {}
    
    # 2. حلقه پردازش تک‌تک ارزها در سبد معاملاتی
    for symbol in SYMBOLS:
        print(f"\n[INFO] ➜ Deep Scanning: {symbol}")
        
        # الف) دانلود دیتای پایه (M15 و H1)
        df_m15 = yf.download(symbol, period="10d", interval="15m", progress=False)
        df_h1 = yf.download(symbol, period="20d", interval="1h", progress=False)
        
        if df_m15.empty or df_h1.empty:
            print(f"[WARNING] No data fetched for {symbol}. Skipped.")
            continue
            
        # یکسان‌سازی نام ستون‌ها (جلوگیری از خطای Multi-Index)
        df_m15.columns = [col[0] if isinstance(col, tuple) else col for col in df_m15.columns]
        df_h1.columns = [col[0] if isinstance(col, tuple) else col for col in df_h1.columns]
        
        # ب) ساخت تایم‌فریم 4 ساعته (H4) با روش Resampling
        # پایتون کندل‌های 1 ساعته را ترکیب می‌کند تا کندل 4 ساعته دقیق بسازد
        df_h4 = df_h1.resample('4H').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        # پ) تحلیل بُعد اول: روند کلان (H4) - دیدگاه بانک‌های مرکزی
        df_h4['ema_50'] = df_h4['Close'].ewm(span=50, adjust=False).mean()
        macro_trend = "buy" if df_h4['Close'].iloc[-1] > df_h4['ema_50'].iloc[-1] else "sell"
        
        # ت) تحلیل بُعد دوم: روند میانی (H1) - تاییدیه جریان سفارشات
        df_h1['ema_50'] = df_h1['Close'].ewm(span=50, adjust=False).mean()
        micro_trend = "buy" if df_h1['Close'].iloc[-1] > df_h1['ema_50'].iloc[-1] else "sell"
        
        # اگر روند 4 ساعته و 1 ساعته هم‌جهت نباشند، بازار پر از نویز است
        mtf_sync = (macro_trend == micro_trend)
        target_trend = macro_trend if mtf_sync else "flat"
        
        # ث) تحلیل بُعد سوم: نقطه ورود دقیق (M15)
        df_m15['vwap'] = calculate_vwap(df_m15)
        df_m15['ema_200'] = df_m15['Close'].ewm(span=200, adjust=False).mean()
        df_m15['rsi'] = calculate_rsi(df_m15['Close'])
        
        # محاسبه ATR برای حد ضررهای داینامیک و منطقی
        tr = pd.concat([df_m15['High'] - df_m15['Low'], 
                        np.abs(df_m15['High'] - df_m15['Close'].shift()), 
                        np.abs(df_m15['Low'] - df_m15['Close'].shift())], axis=1).max(axis=1)
        df_m15['atr'] = tr.rolling(14).mean()

# ج) تشخیص رژیم بازار اختصاصی همین ارز
        hurst = calculate_hurst(df_m15['Close'].tail(100).values)
        market_regime = "trending" if hurst > 0.52 else "range"
        
        # چ) تاییدیه اسمارت مانی (VWAP)
        vwap_current = df_m15['vwap'].iloc[-2]
        close_current = current['Close']
        
        # بانک‌ها در منطقه ارزان می‌خرند و در گران می‌فروشند
        vwap_discount = (close_current <= vwap_current * 1.0005) # قیمت در حوالی یا زیر VWAP باشد
        vwap_premium = (close_current >= vwap_current * 0.9995)  # قیمت در حوالی یا بالای VWAP باشد
        
        # ح) اجرای ماشین لرنینگ اختصاصی با ویژگی‌های حجمی
        ml_prob_up = run_ml_prediction_for_asset(df_m15)
        
        direction = "flat"
        atr_pips = (current['atr'] * 10000)

        # ======================================================================
        # 6. GOD MODE DECISION MATRIX (ماتریس تصمیم‌گیری ۳ بُعدی)
        # ======================================================================
        # الف: استراتژی روند نهادی (Trend Following)
        # فقط در صورتی اجازه ورود داریم که روند 4 ساعته و 1 ساعته هم‌سو باشند (target_trend)
        if target_trend != "flat" and market_regime == "trending":
            bullish_fvg = prev['Low'] > df_m15.iloc[-4]['High']
            bearish_fvg = prev['High'] < df_m15.iloc[-4]['Low']
            
            # شرط خرید: روند کلان صعودی + قیمت بالای EMA200 + گپ نقدینگی + قیمت در منطقه تخفیف VWAP
            if target_trend == "buy" and current['Close'] > current['ema_200'] and bullish_fvg and vwap_discount:
                direction = "buy"
            # شرط فروش: روند کلان نزولی + قیمت زیر EMA200 + گپ نقدینگی + قیمت در منطقه گران VWAP
            elif target_trend == "sell" and current['Close'] < current['ema_200'] and bearish_fvg and vwap_premium:
                direction = "sell"
                
        # ب: استراتژی بازگشت به میانگین با فیلتر حجم (Mean Reversion)
        elif target_trend == "flat" and market_regime == "range":
            bb_width = calculate_bollinger_width(df_m15).iloc[-2]
            # فقط در بازارهایی که فشردگی شدید (Squeeze) ندارند نوسان‌گیری می‌کنیم تا گیر شکست‌های ناگهانی نیفتیم
            if bb_width > 0.002: 
                bb_std = df_m15['Close'].rolling(20).std()
                bb_lower = df_m15['Close'].rolling(20).mean() - (bb_std * 2)
                bb_upper = df_m15['Close'].rolling(20).mean() + (bb_std * 2)
                
                if current['Close'] <= bb_lower.iloc[-2] and current['rsi'] < 35: direction = "buy"
                elif current['Close'] >= bb_upper.iloc[-2] and current['rsi'] > 65: direction = "sell"

        # ======================================================================
        # 7. INSTITUTIONAL VETO FILTERS (فیلترهای نهایی محافظت از سرمایه)
        # ======================================================================
        veto_reason = ""
        currency_code = symbol[:3]
        base_news = news_matrix.get(currency_code, {"sentiment": 0.0, "blocked": False})
        usd_news = news_matrix.get("USD", {"sentiment": 0.0, "blocked": False})
        
        if direction != "flat":
            # 1. فیلتر قوی سیاه (Black Swan Circuit Breaker)
            if base_news["blocked"] or usd_news["blocked"]:
                veto_reason = "CRITICAL NEWS BLOCK (Black Swan)"
                direction = "flat"
                
            # 2. فیلتر هوش مصنوعی حجمی (سخت‌گیری افزایش یافته به 55 درصد)
            elif direction == "buy" and ml_prob_up < 0.55: 
                veto_reason = f"ML Vol-Prob < 55% Up ({ml_prob_up:.1%})"
                direction = "flat"
            elif direction == "sell" and ml_prob_up > 0.45:
                veto_reason = f"ML Vol-Prob < 55% Down ({(1-ml_prob_up):.1%})"
                direction = "flat"
                
            # 3. فیلتر تضاد فاندامنتال (NLP Sentiment Mismatch)
            elif direction == "buy" and (base_news["sentiment"] < -0.3 and usd_news["sentiment"] > 0.3):
                veto_reason = "NLP Mismatch (Bearish News)"
                direction = "flat"
            elif direction == "sell" and (base_news["sentiment"] > 0.3 and usd_news["sentiment"] < -0.3):
                veto_reason = "NLP Mismatch (Bullish News)"
                direction = "flat"

        # ======================================================================
        # 8. DYNAMIC SCALE-OUT TARGETS (محاسبه تارگت‌های پله‌ای)
        # ======================================================================
        # محاسبه استاپ لاس و دو تارگت سود برای خروج پله‌ای (Partial Close)
        sl_pips = round(atr_pips * 1.5, 1)
        tp1_pips = round(atr_pips * 1.5, 1) # تارگت اول: ریسک به ریوارد 1:1 (سیو سود نصف حجم)
        # تارگت دوم: شکار کل روند تا 4 برابر ریسک!
        tp2_pips = round(atr_pips * 4.0, 1) if market_regime == "trending" else round(atr_pips * 2.5, 1) 
        
        # ذخیره در دیکشنری نهایی برای ارسال به متاتریدر
        portfolio_results[symbol] = {
            "regime": market_regime,
            "direction": direction,
            "target_tp1_pips": max(tp1_pips, 10.0),
            "target_tp2_pips": max(tp2_pips, 20.0),
            "target_sl_pips": max(sl_pips, 12.0),
            "risk_percent": RISK_PER_TRADE_PERCENT, # ارسال درصد ریسک به جای حجم ثابت
            "veto": veto_reason
        }
# ======================================================================
    # 9. MASTER JSON OUTPUT & TELEGRAM REPORTING (خروجی و گزارش‌گیری کلان)
    # ======================================================================
    # اکنون از حلقه for (پردازش ارزها) خارج شده‌ایم
    
    # ساختار نهایی و سازمانی فایل JSON برای خوانش هم‌زمان 5 چارت در متاتریدر
    output_data = {
        "last_update": str(datetime.datetime.now()),
        "global_risk_percent": RISK_PER_TRADE_PERCENT, # درصد ریسک برای مدیریت سرمایه
        "assets": portfolio_results
    }
    
    # ذخیره فایل در گیت‌هاب
    with open("sentinel_config.json", "w") as f:
        json.dump(output_data, f, indent=4)
        
    # ساخت گزارش شگفت‌انگیز و حرفه‌ای برای تلگرام (نمای صندوق سرمایه‌گذاری)
    msg = f"<b>🏛 Sentinel God Mode (V6)</b>\n"
    msg += f"⏱ {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
    msg += f"⚖️ Risk Profile: {RISK_PER_TRADE_PERCENT}% per trade\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
    
    for sym, data in portfolio_results.items():
        clean_sym = sym.replace("=X", "") # پاک کردن =X برای زیبایی
        
        if data['direction'] != "flat":
            # ارزهایی که سیگنال ورود دارند
            msg += f"🟢 <b>{clean_sym}</b>: <b>{data['direction'].upper()}</b>\n"
            msg += f"   🎯 TP1: {data['target_tp1_pips']} | TP2: {data['target_tp2_pips']}\n"
            msg += f"   🛡 SL: {data['target_sl_pips']}\n"
        else:
            # ارزهایی که در استراحت هستند یا وتو شده‌اند
            reason = data['veto'] if data['veto'] else "Awaiting Setup"
            msg += f"⚪️ <b>{clean_sym}</b>: FLAT\n"
            msg += f"   ⏳ <i>{reason}</i>\n"
            
        msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
            
    send_telegram_alert(msg)
    print("\n[SUCCESS] God Mode Execution Completed. Master JSON generated and Telegram sent.")

# ======================================================================
# اجرای اصلی برنامه
# ======================================================================
if __name__ == "__main__":
    generate_god_mode_strategy()
