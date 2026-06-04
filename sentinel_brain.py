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

nltk.download('vader_lexicon', quiet=True)

# ======================================================================
# 1. INSTITUTIONAL CONFIGURATION (تنظیمات سطح سازمان)
# ======================================================================
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE" 
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"     

SYMBOLS = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "USDJPY=X"]

NORMAL_RISK_PERCENT = 2.0 
CRISIS_RISK_PERCENT = 1.0 

def send_telegram_alert(message):
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE": return 
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try: requests.post(url, data=payload, timeout=5)
    except: pass

# ======================================================================
# 2. PRICE ACTION & QUANTITATIVE INDICATORS (اندیکاتورها و پرایس اکشن)
# ======================================================================
def calculate_vwap(df):
    q = df['Volume'].replace(0, 1) 
    p = (df['High'] + df['Low'] + df['Close']) / 3
    return (p * q).cumsum() / q.cumsum()

def calculate_hurst(price_series, max_lag=20):
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
    std = df['Close'].rolling(window=window).std()
    ma = df['Close'].rolling(window=window).mean()
    upper = ma + (std * 2)
    lower = ma - (std * 2)
    return (upper - lower) / ma

def find_major_swings(df, lookback=50):
    """
    پیدا کردن سقف و کف‌های ماژور (Swing High / Swing Low) 
    برای رسم فیبوناچی و تعیین استاپ‌لاس‌های منطقی
    """
    if len(df) < lookback: return None, None
    
    # پیدا کردن بالاترین و پایین‌ترین نقطه در 50 کندل گذشته
    recent_data = df.iloc[-lookback:-2] # 50 کندل قبل تا کندل تایید شده آخر
    swing_high = recent_data['High'].max()
    swing_low = recent_data['Low'].min()
    
    return swing_high, swing_low

def check_liquidity_sweep(df, lookback=20):
    if len(df) < lookback + 2: return "none"
    recent_high = df['High'].iloc[-lookback-2:-2].max()
    recent_low = df['Low'].iloc[-lookback-2:-2].min()
    current = df.iloc[-2] 
    
    if current['High'] > recent_high and current['Close'] < recent_high:
        return "bearish_sweep"
    elif current['Low'] < recent_low and current['Close'] > recent_low:
        return "bullish_sweep"
    return "none"

# ======================================================================
# 3. CRISIS NLP NEWS ENGINE (موتور تشخیص فاندامنتال و آلفای بحران)
# ======================================================================
def analyze_crisis_news():
    """
    استفاده از الگوریتم پیشرفته VADER برای درک لحن اخبار اقتصادی.
    اگر بحرانی رخ دهد، ربات خاموش نمی‌شود، بلکه نوع بحران را شناسایی کرده 
    و سیگنال Crisis Mode صادر می‌کند تا استراتژی‌های شکار نقدینگی فعال شوند.
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
    
    # ذخیره وضعیت بحران برای هر ارز به طور مستقل
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
                        
                        # جستجو برای کشف بحران‌های پول‌ساز
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
# 4. INSTITUTIONAL ORDER BLOCK & STRUCTURE ENGINE (موتور اوردر بلاک)
# ======================================================================
def detect_institutional_order_blocks(df, lookback=50):
    """
    کشف پویا و پیشرفته اوردر بلاک‌های سازمانی (Order Blocks) 
    و تعیین محدوده‌های ارزان‌فروشی و گران‌فروشی بانک‌ها
    """
    if len(df) < lookback:
        return None
        
    recent = df.iloc[-lookback:-1].copy()
    
    # 1. پیدا کردن سقف و کف محدوده معاملاتی زنده (Dealing Range)
    dealing_high = recent['High'].max()
    dealing_low = recent['Low'].min()
    equilibrium = (dealing_high + dealing_low) / 2.0
    
    order_blocks = {
        "bullish_ob_top": 0.0, "bullish_ob_bottom": 0.0, "bullish_valid": False,
        "bearish_ob_top": 0.0, "bearish_ob_bottom": 0.0, "bearish_valid": False,
        "equilibrium": equilibrium, "dealing_high": dealing_high, "dealing_low": dealing_low
    }
    
    # 2. الگوریتم کشف اوردر بلاک صعودی (آخرین کندل نزولی قبل از رالی صعودی شدید)
    # ما به دنبال کندلی می‌گردیم که بعد از آن قیمت پرواز کرده و سقف قبلی را شکسته (BOS)
    for i in range(len(recent) - 5, 2, -1):
        cond_down_candle = recent['Close'].iloc[i] < recent['Open'].iloc[i]
        cond_strong_move = recent['Close'].iloc[i+1] > recent['High'].iloc[i] and recent['Close'].iloc[i+2] > recent['Close'].iloc[i+1]
        
        if cond_down_candle and cond_strong_move:
            # اوردر بلاک صعودی باید در منطقه تخفیف (زیر Equilibrium) باشد
            if recent['High'].iloc[i] < equilibrium:
                order_blocks["bullish_ob_top"] = recent['High'].iloc[i]
                order_blocks["bullish_ob_bottom"] = recent['Low'].iloc[i]
                order_blocks["bullish_valid"] = True
                break
                
    # 3. الگوریتم کشف اوردر بلاک نزولی (آخرین کندل صعودی قبل از ریزش شدید بانک‌ها)
    for i in range(len(recent) - 5, 2, -1):
        cond_up_candle = recent['Close'].iloc[i] > recent['Open'].iloc[i]
        cond_sharp_drop = recent['Close'].iloc[i+1] < recent['Low'].iloc[i] and recent['Close'].iloc[i+2] < recent['Close'].iloc[i+1]
        
        if cond_up_candle and cond_sharp_drop:
            # اوردر بلاک نزولی باید در منطقه گران (بالای Equilibrium) باشد
            if recent['Low'].iloc[i] > equilibrium:
                order_blocks["bearish_ob_top"] = recent['High'].iloc[i]
                order_blocks["bearish_ob_bottom"] = recent['Low'].iloc[i]
                order_blocks["bearish_valid"] = True
                break
                
    return order_blocks

# ======================================================================
# 5. MULTI-TIMEFRAME QUANT ENGINE (موتور تحلیل ۳ بُعدی زنده)
# ======================================================================
def run_ml_prediction_for_asset(df):
    """ آموزش هوش مصنوعی روی متغیرهای نقدینگی و ساختار بازار """
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

# استخراج اوردر بلاک‌ها و مناطق تعادلی بانک‌ها (SMC)
        ob_data = detect_institutional_order_blocks(df_m15)
        if not ob_data: continue
        
        # اجرای ماشین لرنینگ برای تاییدیه نهایی
        ml_prob_up = run_ml_prediction_for_asset(df_m15)
        
        direction = "flat"
        atr_pips = (current['atr'] * 10000)
        price_current = current['Close']
        
        # استاپ‌لاس پیش‌فرض (اگر سیگنالی صادر نشد بر اساس ATR می‌ماند)
        sl_pips = round(atr_pips * 1.5, 1)

        currency_code = symbol[:3]
        base_news = news_matrix.get(currency_code, {"sentiment": 0.0, "crisis_mode": False, "crisis_type": "none"})
        usd_news = news_matrix.get("USD", {"sentiment": 0.0, "crisis_mode": False, "crisis_type": "none"})
        
        is_crisis = base_news["crisis_mode"] or usd_news["crisis_mode"]
        active_risk = CRISIS_RISK_PERCENT if is_crisis else NORMAL_RISK_PERCENT

        # بررسی نفوذ قیمت به اوردر بلاک‌ها (Mitigation)
        # آیا قیمت زنده الان دقیقاً داخل اوردر بلاک بانک‌هاست؟
        testing_bullish_ob = ob_data["bullish_valid"] and (ob_data["bullish_ob_bottom"] <= price_current <= ob_data["bullish_ob_top"])
        testing_bearish_ob = ob_data["bearish_valid"] and (ob_data["bearish_ob_bottom"] <= price_current <= ob_data["bearish_ob_top"])

        # ======================================================================
        # 6. ULTIMATE SNIPER DECISION MATRIX (ماتریس نقطه‌زنی)
        # ======================================================================
        if is_crisis:
            # 🔴 استراتژی بحران: شکار استاپ‌لاس خرد (Liquidity Sweep)
            if liquidity_sweep == "bullish_sweep" and target_trend != "sell":
                direction = "buy"
                sl_pips = round(atr_pips * 2.5, 1) # استاپ گشادتر برای در امان ماندن از شدوهای جنگ
            elif liquidity_sweep == "bearish_sweep" and target_trend != "buy":
                direction = "sell"
                sl_pips = round(atr_pips * 2.5, 1)
        else:
            # 🟢 استراتژی روزهای عادی: پرایس اکشن سازمانی (SMC Mitigation)
            if target_trend == "buy" and testing_bullish_ob:
                # ورود در ارزان‌فروشی! روند کلان صعودی است و قیمت به اوردر بلاک تخفیف‌دار رسیده
                direction = "buy"
                # حد ضرر هوشمند: دقیقاً زیر اوردر بلاک یا زیر آخرین کف (Dealing Low) + 2 پیپ بافر
                sl_price = min(ob_data["bullish_ob_bottom"], ob_data["dealing_low"])
                sl_pips = max(abs(price_current - sl_price) * 10000 + 2.0, 10.0) # حداقل 10 پیپ برای اسپرد
                
            elif target_trend == "sell" and testing_bearish_ob:
                # ورود در گران‌فروشی! روند کلان نزولی است و قیمت به اوردر بلاک گران رسیده
                direction = "sell"
                # حد ضرر هوشمند: دقیقاً بالای اوردر بلاک یا بالای سقف قبلی + 2 پیپ بافر
                sl_price = max(ob_data["bearish_ob_top"], ob_data["dealing_high"])
                sl_pips = max(abs(sl_price - price_current) * 10000 + 2.0, 10.0)

        # ======================================================================
        # 7. INSTITUTIONAL VETO FILTERS (فیلترهای نهایی تطبیقی)
        # ======================================================================
        veto_reason = ""
        
        if direction != "flat":
            if not is_crisis:
                if direction == "buy" and ml_prob_up < 0.55: 
                    veto_reason = f"ML Vol-Prob < 55% Up ({ml_prob_up:.1%})"
                    direction = "flat"
                elif direction == "sell" and ml_prob_up > 0.45:
                    veto_reason = f"ML Vol-Prob < 55% Down ({(1-ml_prob_up):.1%})"
                    direction = "flat"
                    
                # فیلتر تاییدیه جریان سفارش (VWAP)
                # در SMC، حتی اگر اوردر بلاک هم بود، قیمت باید با VWAP هم‌خوانی داشته باشد
                elif direction == "buy" and price_current > current['vwap']:
                    veto_reason = "Price above VWAP (Not Discounted Enough)"
                    direction = "flat"
                elif direction == "sell" and price_current < current['vwap']:
                    veto_reason = "Price below VWAP (Not Premium Enough)"
                    direction = "flat"
            else:
                # در زمان بحران، فقط اگر هوش مصنوعی کاملاً مخالف بود وتو می‌کنیم
                if direction == "buy" and ml_prob_up < 0.40:
                    veto_reason = "Crisis ML Veto"
                    direction = "flat"
                elif direction == "sell" and ml_prob_up > 0.60:
                    veto_reason = "Crisis ML Veto"
                    direction = "flat"

        # ======================================================================
        # 8. DYNAMIC ASYMMETRIC TARGETS (تارگت‌های نامتقارن هوشمند)
        # ======================================================================
        if direction != "flat":
            if is_crisis:
                tp1_pips = round(sl_pips * 1.5, 1) # سیو سود اولیه در بحران
                tp2_pips = round(sl_pips * 5.0, 1) # شکار 5 برابر استاپ‌لاس در حرکت شارپ
            else:
                # در SMC ریسک به ریواردها به شدت بالاست (چون استاپ‌ها بسیار کوچک و دقیق هستند)
                tp1_pips = round(sl_pips * 2.0, 1) # تارگت اول = دو برابر استاپ لاس
                # تارگت دوم = سقف یا کف مقابل (Liquidity Target)
                if direction == "buy":
                    tp2_pips = max(abs(ob_data["dealing_high"] - price_current) * 10000, tp1_pips * 2)
                else:
                    tp2_pips = max(abs(price_current - ob_data["dealing_low"]) * 10000, tp1_pips * 2)
        else:
            tp1_pips = 15.0
            tp2_pips = 30.0

        # بسته‌بندی در دیکشنری برای این جفت‌ارز
        portfolio_results[symbol] = {
            "regime": market_regime,
            "direction": direction,
            "target_tp1_pips": round(tp1_pips, 1),
            "target_tp2_pips": round(tp2_pips, 1),
            "target_sl_pips": round(sl_pips, 1),
            "risk_percent": active_risk,
            "is_crisis": is_crisis,
            "veto": veto_reason
        }

# ======================================================================
    # 9. MASTER JSON OUTPUT & TELEGRAM REPORTING (خروجی و گزارش‌گیری کلان)
    # ======================================================================
    # اکنون از حلقه for (پردازش 5 ارز) خارج شده‌ایم
    
    # ساختار نهایی فایل JSON با معماری SMC
    output_data = {
        "last_update": str(datetime.datetime.utcnow()) + " UTC",
        "global_strategy": "SMC_OrderBlock_Mitigation",
        "assets": portfolio_results
    }
    
    # ذخیره فایل برای خوانش توسط متاتریدر
    with open("sentinel_config.json", "w") as f:
        json.dump(output_data, f, indent=4)
        
    # ساخت گزارش داشبورد فوق‌حرفه‌ای برای تلگرام
    msg = f"<b>🏛 Sentinel V8 (SMC Sniper)</b>\n"
    msg += f"⏱ {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
    
    for sym, data in portfolio_results.items():
        clean_sym = sym.replace("=X", "") 
        
        # برچسب‌گذاری گرافیکی برای تمایز روزهای عادی (SMC) از روزهای جنگی (Crisis)
        mode_icon = "🚨 CRISIS" if data.get('is_crisis', False) else "🛡 SMC"
        
        if data['direction'] != "flat":
            # ارزهایی که دقیقاً به اوردر بلاک رسیده‌اند و سیگنال ورود دارند
            msg += f"🟢 <b>{clean_sym}</b>: <b>{data['direction'].upper()}</b> [{mode_icon}]\n"
            msg += f"   🎯 TP1: {data['target_tp1_pips']} | TP2: {data['target_tp2_pips']}\n"
            msg += f"   🛑 SL: {data['target_sl_pips']} | ⚖️ Risk: {data['risk_percent']}%\n"
        else:
            # ارزهایی که منتظر رسیدن قیمت به اوردر بلاک هستند یا وتو شده‌اند
            reason = data['veto'] if data['veto'] else "Awaiting OB Mitigation"
            msg += f"⚪️ <b>{clean_sym}</b>: FLAT [{mode_icon}]\n"
            msg += f"   ⏳ <i>{reason}</i>\n"
            
        msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
            
    send_telegram_alert(msg)
    print("\n[SUCCESS] V8 SMC Sniper Execution Completed. Master JSON generated and Telegram sent.")

# ======================================================================
# نقطه شروع اجرای برنامه (Main Entry Point)
# ======================================================================
if __name__ == "__main__":
    generate_god_mode_strategy()
