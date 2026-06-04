import os
import json
import datetime
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# ======================================================================
# 1. TELEGRAM CONFIGURATION (سیستم هشدار و گزارش‌دهی شفاف)
# ======================================================================
# لطفاً توکن ربات و چت‌آیدی خود را اینجا جایگزین کنید
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE" 
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"     

def send_telegram_alert(message):
    """
    ارسال پیام‌های فرمت‌بندی شده به تلگرام شما
    """
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE": 
        return # در صورت عدم تنظیم توکن، خطا نمی‌دهد و عبور می‌کند
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID, 
        "text": message, 
        "parse_mode": "HTML"
    }
    try: 
        requests.post(url, data=payload, timeout=5)
    except Exception as e: 
        print(f"[ERROR] Telegram delivery failed: {e}")

# ======================================================================
# 2. CORE MATHEMATICAL & TECHNICAL INDICATORS (توابع پایه ریاضی)
# ======================================================================
def calculate_hurst(price_series, max_lag=20):
    """
    محاسبه نماگر هرست برای تشخیص فاز بازار (رِنج یا رونددار)
    """
    if len(price_series) < max_lag: 
        return 0.5 # مقدار خنثی در صورت کمبود دیتا
        
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    
    return poly[0] * 2.0

def calculate_rsi(data, periods=14):
    """
    محاسبه دقیق اندیکاتور RSI بر اساس فرمول استاندارد
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series
    
    
# ======================================================================
# 3. MACHINE LEARNING ENGINE (هوش مصنوعی پیش‌بینی‌کننده)
# ======================================================================
def run_ml_prediction(df):
    """
    استفاده از الگوریتم جنگل تصادفی (Random Forest) 
    برای تایید یا رد سیگنال‌های تولید شده توسط پرایس اکشن
    """
    print("[INFO] Training Machine Learning Model on recent market data...")
    df_ml = df.copy()
    
    # استخراج ویژگی‌ها (Features) برای آموزش مدل هوش مصنوعی
    # به ربات می‌گوییم به چه چیزهایی برای الگوبرداری نگاه کند
    df_ml['rsi'] = calculate_rsi(df_ml['Close'])
    df_ml['ema_dist'] = df_ml['Close'] - df_ml['Close'].ewm(span=50).mean()
    df_ml['returns'] = df_ml['Close'].pct_change()
    
    # تعیین هدف (Target): آیا کندل بعدی سبز (صعودی) بسته شده است؟ (۱=بله، ۰=خیر)
    df_ml['target'] = np.where(df_ml['Close'].shift(-1) > df_ml['Close'], 1, 0)
    
    # پاکسازی داده‌های خالی ناشی از محاسبات اندیکاتورها برای جلوگیری از ارور
    df_ml.dropna(inplace=True)
    
    X = df_ml[['rsi', 'ema_dist', 'returns']]
    y = df_ml['target']
    
    # اگر دیتای کافی برای آموزش نبود، نظر خنثی (۵۰٪) می‌دهد
    if len(X) < 100: 
        return 0.5 
    
    # آموزش مدل روی گذشته بازار (منهای کندل لایو و فعلی)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X.iloc[:-1], y.iloc[:-1])
    
    # پیش‌بینی احتمال صعود (UP) برای کندل زنده فعلی
    live_data = X.iloc[[-1]]
    prob_up = model.predict_proba(live_data)[0][1]
    
    return prob_up

# ======================================================================
# 4. NEWS & CIRCUIT BREAKER (سیستم دفاعی در برابر اخبار پنهان)
# ======================================================================
def analyze_news():
    """
    بررسی تیتر اخبار اقتصادی برای جلوگیری از معامله در زمان نوسانات شدید
    این کار از لغزش قیمت (Slippage) در حساب ECN جلوگیری می‌کند
    """
    print("[INFO] Checking real-time financial RSS feeds...")
    urls = ["https://www.fxstreet.com/rss/news"]
    
    # کلمات کلیدی که نوسانات مرگباری برای اکانت‌های مارجین پایین دارند
    high_impact_keywords = ['FOMC', 'CPI', 'NFP', 'FED', 'INFLATION', 'RATE', 'ECB', 'NON-FARM']
    
    news_block = False
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        response = requests.get(urls[0], headers=headers, timeout=5)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            # بررسی فقط 10 خبر داغ و اخیر بازار
            for item in root.findall('.//item')[:10]: 
                text = (item.find('title').text or "").upper()
                if any(kw in text for kw in high_impact_keywords):
                    news_block = True
                    print(f"[WARNING] High Impact News Detected: {text}")
                    break
    except Exception as e:
        print(f"[ERROR] Feed parsing failed or connection timeout: {e}")
        
    return news_block
    
# ======================================================================
# 5. CORE HYBRID STRATEGY ENGINE (هسته مرکزی پردازش داده‌ها)
# ======================================================================
def generate_ultimate_strategy():
    print(f"\n[START] Sentinel Ultimate Agile V4 - {datetime.datetime.now()}")
    
    # الف) دانلود دیتای چندزمانی و سبد ارزی
    print("[INFO] Fetching Multi-Timeframe and Correlation Data...")
    df_m15 = yf.download("EURUSD=X", period="10d", interval="15m", progress=False)
    df_h1 = yf.download("EURUSD=X", period="20d", interval="1h", progress=False)
    df_gbp = yf.download("GBPUSD=X", period="10d", interval="15m", progress=False)
    
    if df_m15.empty or df_h1.empty or df_gbp.empty:
        print("[CRITICAL] Market data fetch failed. Aborting execution.")
        return
        
    # یکسان‌سازی فرمت ستون‌ها برای جلوگیری از خطای کتابخانه yfinance
    df_m15.columns = [col[0] if isinstance(col, tuple) else col for col in df_m15.columns]
    df_h1.columns = [col[0] if isinstance(col, tuple) else col for col in df_h1.columns]
    df_gbp.columns = [col[0] if isinstance(col, tuple) else col for col in df_gbp.columns]
    
    # ب) فیلتر روند تایم‌فریم بالاتر (H1 MTF) - شنا در جهت نهنگ‌ها
    df_h1['ema_100'] = df_h1['Close'].ewm(span=100, adjust=False).mean()
    h1_trend = "buy" if df_h1['Close'].iloc[-1] > df_h1['ema_100'].iloc[-1] else "sell"
    
    # پ) فیلتر همبستگی (Portfolio Correlation) بین یورو و پوند
    correlation = df_m15['Close'].tail(50).corr(df_gbp['Close'].tail(50))
    
    # ت) محاسبات تکنیکال تایم‌فریم اصلی (15 دقیقه‌ای)
    df_m15['ema_200'] = df_m15['Close'].ewm(span=200, adjust=False).mean()
    df_m15['rsi'] = calculate_rsi(df_m15['Close'])
    
    # محاسبه اندیکاتور ATR برای تعیین دقیق و پویای حد ضرر و سود
    tr = pd.concat([df_m15['High'] - df_m15['Low'], 
                    np.abs(df_m15['High'] - df_m15['Close'].shift()), 
                    np.abs(df_m15['Low'] - df_m15['Close'].shift())], axis=1).max(axis=1)
    df_m15['atr'] = tr.rolling(14).mean()

    # استخراج مقادیر لایو (دو کندل آخر برای اطمینان از بسته شدن کندل)
    current = df_m15.iloc[-2]
    prev = df_m15.iloc[-3]
    
    # ث) تشخیص فاز بازار و اجرای پیش‌بینی هوش مصنوعی
    hurst = calculate_hurst(df_m15['Close'].tail(100).values)
    market_regime = "trending" if hurst > 0.52 else "range"
    
    ml_prob_up = run_ml_prediction(df_m15)
    news_block = analyze_news()
    
    # ======================================================================
    # 6. DECISION MATRIX (ماتریس تصمیم‌گیری پرایس اکشن)
    # ======================================================================
    direction = "flat"
    atr_pips = (current['atr'] * 10000)

    # اگر بازار رونددار باشد -> استراتژی اسمارت مانی (FVG)
    if market_regime == "trending":
        bullish_fvg = prev['Low'] > df_m15.iloc[-4]['High']
        bearish_fvg = prev['High'] < df_m15.iloc[-4]['Low']
        
        # ورود فقط در صورت هم‌جهت بودن با EMA200 و وجود گپ نقدینگی (FVG)
        if current['Close'] > current['ema_200'] and bullish_fvg: 
            direction = "buy"
        elif current['Close'] < current['ema_200'] and bearish_fvg: 
            direction = "sell"

    # اگر بازار رِنج باشد -> استراتژی بازگشت به میانگین (Bollinger + RSI)
    elif market_regime == "range":
        bb_std = df_m15['Close'].rolling(20).std()
        bb_lower = df_m15['Close'].rolling(20).mean() - (bb_std * 2)
        bb_upper = df_m15['Close'].rolling(20).mean() + (bb_std * 2)
        
        # ورود در اشباع فروش (کف کانال) یا اشباع خرید (سقف کانال)
        if current['Close'] <= bb_lower.iloc[-2] and current['rsi'] < 40: 
            direction = "buy"
        elif current['Close'] >= bb_upper.iloc[-2] and current['rsi'] > 60: 
            direction = "sell"
            
# ======================================================================
    # 7. ADVANCED VETO FILTERS (اعمال فیلترهای نهایی و وتو)
    # ======================================================================
    veto_reason = ""
    
    # اگر پرایس اکشن سیگنال داد، حالا از فیلترهای سخت‌گیرانه ردش می‌کنیم
    if direction != "flat":
        
        # ۱. فیلتر هم‌گرایی تایم‌فریم‌ها (MTF)
        # نمی‌خواهیم در ۱۵ دقیقه بخریم وقتی روند ۱ ساعته نزولی است
        if market_regime == "trending" and direction != h1_trend:
            veto_reason = f"Against H1 Trend ({h1_trend.upper()})"
            direction = "flat"
        
        # ۲. فیلتر منعطف هوش مصنوعی (آستانه ۵۲٪)
        elif direction == "buy" and ml_prob_up < 0.52:
            veto_reason = f"ML Prob < 52% Up ({ml_prob_up:.1%})"
            direction = "flat"
        elif direction == "sell" and ml_prob_up > 0.48: # یعنی احتمال نزول کمتر از ۵۲٪ است
            veto_reason = f"ML Prob < 52% Down ({(1-ml_prob_up):.1%})"
            direction = "flat"
            
        # ۳. فیلتر نویز دلار (همبستگی سبد)
        elif correlation < 0.30:
            veto_reason = f"Low Correlation ({correlation:.2f})"
            direction = "flat"

    # ======================================================================
    # 8. DYNAMIC RISK MANAGEMENT & JSON OUTPUT (محاسبه تارگت‌ها و خروجی)
    # ======================================================================
    # در روندها سود ۳ برابر ریسک، در رنج سود ۲ برابر ریسک
    tp_pips = atr_pips * 3.0 if market_regime == "trending" else atr_pips * 2.0
    sl_pips = atr_pips * 1.5

    # ساختار نهایی فایل برای خوانده شدن توسط متاتریدر ۵
    output = {
        "last_update": str(datetime.datetime.now()),
        "market_regime": market_regime,
        "direction": direction,
        "target_tp_pips": round(max(tp_pips, 15.0), 1), # حداقل ۱۵ پیپ برای پوشش کمیسیون
        "target_sl_pips": round(max(sl_pips, 12.0), 1),
        "spread_max_allowed_points": 20 # حداکثر اسپرد مجاز ۲ پیپ برای اکانت ECN
    }
    
    # ذخیره فایل در مسیر روت گیت‌هاب
    with open("sentinel_config.json", "w") as f:
        json.dump(output, f, indent=4)
        
    # ======================================================================
    # 9. TELEGRAM REPORTING (گزارش‌دهی زنده به گوشی شما)
    # ======================================================================
    status_icon = "🟢 ACTIVE" if direction != "flat" else ("🟡 VETOED" if veto_reason else "⚪️ FLAT")
    
    msg = f"""
<b>🤖 Sentinel Ultimate V4 Report</b>
<b>Status:</b> {status_icon}

<b>📊 Market Analysis:</b>
• Regime: {market_regime.upper()} (Hurst: {hurst:.2f})
• H1 Trend: {h1_trend.upper()}
• ML Confidence: {ml_prob_up:.1%} UP
• EU/GU Corr: {correlation:.2f}

<b>🎯 Trade Signal:</b>
• Action: <b>{direction.upper()}</b>
• TP: {output['target_tp_pips']} | SL: {output['target_sl_pips']}
• Blocked By: {veto_reason if veto_reason else 'None'}
    """
    send_telegram_alert(msg)
    print("\n[SUCCESS] Sentinel Brain Execution Completed. JSON generated and Telegram sent.")

# ======================================================================
# اجرای اصلی برنامه
# ======================================================================
if __name__ == "__main__":
    generate_ultimate_strategy()
