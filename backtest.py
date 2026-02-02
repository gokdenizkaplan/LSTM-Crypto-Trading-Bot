import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import datetime

# --- 🚀 HODL+ (YAPIŞKAN HİBRİT) STRATEJİSİ ---
SEMBOL = "BTC-USD"
MODEL_DOSYASI = "sampiyon_model.h5"
SCALER_DOSYASI = "sampiyon_scaler.gz"

KOMISYON = 0.001
BASLANGIC_BAKIYE = 1000

FEATURE_LIST = ['Log_Ret', 'MFI_14', 'NATR_14', 'RSI_14', 'Dist_EMA', 'ROC_10']
LOOK_BACK_DAYS = 30
BUGUN = datetime.now().strftime('%Y-%m-%d')

print(f"--- FİNAL STRATEJİ: HODL+ (Boğada Yapış, Ayıda Kaç) ---")

# 1. Yükleme
try:
    model = load_model(MODEL_DOSYASI)
    scaler = joblib.load(SCALER_DOSYASI)
    print("✅ Sistem Hazır.")
except:
    print("❌ Dosyalar eksik.")
    exit()

# 2. Veri
df = yf.download(SEMBOL, start="2019-06-01", end=BUGUN, interval="1d", progress=False)
if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

# 3. İndikatörler
df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
df['MFI_14'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
df['NATR_14'] = ta.natr(df['High'], df['Low'], df['Close'], length=14)
df['RSI_14'] = ta.rsi(df['Close'], length=14)
df['EMA50'] = ta.ema(df['Close'], length=50)
df['Dist_EMA'] = (df['Close'] - df['EMA50']) / df['EMA50']
df['ROC_10'] = ta.roc(df['Close'], length=10)

# --- REJİM VE SİNYAL AYARLARI ---
df['EMA_200'] = ta.ema(df['Close'], length=200)  # Rejim Belirleyici

# GİRİŞLER
df['EMA_FAST'] = ta.ema(df['Close'], length=8)  # Boğa Girişi (Çok Hızlı)
df['EMA_SAFE'] = ta.ema(df['Close'], length=20)  # Ayı Girişi (Güvenli)

# ÇIKIŞLAR (Kritik Değişiklik Burada)
df['EMA_HODL'] = ta.ema(df['Close'], length=100)  # Boğa Çıkışı (Çok Gevşek - Silkelenmez)
df['EMA_PANIC'] = ta.ema(df['Close'], length=20)  # Ayı Çıkışı (Çok Sıkı - Hemen Kaç)

df.dropna(inplace=True)
df = df[df.index >= '2020-01-01']

scaled_data = scaler.transform(df[FEATURE_LIST].values)

# 4. MOTOR
usdt_balance = BASLANGIC_BAKIYE
btc_balance = 0
in_position = False
entry_price = 0

portfolio_history = []
trade_log = []

for i in range(LOOK_BACK_DAYS, len(df) - 1):
    current_price = df['Close'].iloc[i]
    date = df.index[i]
    ema200 = df['EMA_200'].iloc[i]

    # Model Tahmini
    X_window = scaled_data[i - LOOK_BACK_DAYS:i].reshape(1, LOOK_BACK_DAYS, len(FEATURE_LIST))
    prob = model.predict(X_window, verbose=0)[0][0]

    # --- 🧠 REJİM KARARI ---
    is_bull = current_price > ema200

    if is_bull:
        mode = "BOĞA 🐂"
        entry_ma = df['EMA_FAST'].iloc[i]
        exit_ma = df['EMA_HODL'].iloc[i]  # EMA 100 kullanıyoruz (HODL gibi davran)
        ai_panic_limit = 0.0  # İPTAL! Boğada AI korksa bile satma.
        entry_threshold = 0.35  # Giriş kolay
    else:
        mode = "AYI 🐻"
        entry_ma = df['EMA_SAFE'].iloc[i]
        exit_ma = df['EMA_PANIC'].iloc[i]  # EMA 20 kullanıyoruz (Hemen kaç)
        ai_panic_limit = 0.30  # Ayıda AI korkarsa sat
        entry_threshold = 0.60  # Giriş zor

    val = usdt_balance + (btc_balance * current_price)
    portfolio_history.append(val)

    # --- İŞLEM ---
    if in_position:
        should_sell = False
        reason = ""

        # 1. Stop Loss (EMA Kırılımı)
        if current_price < exit_ma:
            should_sell = True
            reason = f"Trend Bitti ({mode})"

        # 2. AI Panik (Sadece Ayıda Aktif)
        elif prob < ai_panic_limit:
            should_sell = True
            reason = "AI Panik"

        if should_sell:
            usdt_balance = (btc_balance * current_price) * (1 - KOMISYON)
            btc_balance = 0
            pnl = (current_price - entry_price) / entry_price
            trade_log.append(f"SAT: {date.date()} | PNL: %{pnl * 100:.1f} | {reason}")
            in_position = False

    else:
        if (prob > entry_threshold) and (current_price > entry_ma):
            btc_balance = (usdt_balance / current_price) * (1 - KOMISYON)
            usdt_balance = 0
            entry_price = current_price
            trade_log.append(f"AL : {date.date()} | Mod: {mode}")
            in_position = True

# Sonuç
last_val = usdt_balance + (btc_balance * df['Close'].iloc[-1])
portfolio_history.append(last_val)

dates = df.index[LOOK_BACK_DAYS:LOOK_BACK_DAYS + len(portfolio_history)]
perf_df = pd.DataFrame(
    {'Bot': portfolio_history, 'HODL': df['Close'].iloc[LOOK_BACK_DAYS:LOOK_BACK_DAYS + len(portfolio_history)].values},
    index=dates)

# HODL Normalize
perf_df['HODL'] = perf_df['HODL'] / perf_df['HODL'].iloc[0] * BASLANGIC_BAKIYE

total_bot = ((portfolio_history[-1] - BASLANGIC_BAKIYE) / BASLANGIC_BAKIYE) * 100
total_hodl = ((perf_df['HODL'].iloc[-1] - BASLANGIC_BAKIYE) / BASLANGIC_BAKIYE) * 100

print("\n" + "=" * 50)
print(f"HODL+ STRATEJİ SONUCU")
print("-" * 50)
print(f"BOT GETİRİSİ : % {total_bot:.2f}")
print(f"HODL GETİRİSİ: % {total_hodl:.2f}")
print("-" * 50)
print(f"DURUM: {'👑 HODL YIKILDI' if total_bot > total_hodl else '⚠️ YAKLAŞTIK AMA YETMEDİ'}")

plt.figure(figsize=(12, 6))
plt.yscale('log')
plt.plot(perf_df.index, perf_df['HODL'], label='HODL', color='gray', alpha=0.5)
plt.plot(perf_df.index, perf_df['Bot'], label='HODL+ Bot', color='gold', linewidth=2)
plt.title(f'Final Kapışma: Bot (%{total_bot:.0f}) vs HODL (%{total_hodl:.0f})')
plt.legend()
plt.show()

# --- İŞLEM ANALİZİ RAPORU ---
print("\n" + "=" * 40)
print("📊 DETAYLI İŞLEM RAPORU")
print("-" * 40)

toplam_islem = len(trade_log) // 2  # Al ve Sat 1 işlem sayılır
kazancli_islem = 0
zararli_islem = 0
toplam_kar = 0
toplam_zarar = 0

print("İŞLEM GEÇMİŞİ:")
for log in trade_log:
    print(log)  # Her bir işlemi ekrana yazdıralım

    if "SAT" in log:
        # PNL değerini metinden ayıklama (Örn: PNL: %12.50)
        parca = log.split("PNL: %")[1]
        yuzde = float(parca.split(" ")[0])  # Sadece sayıyı al

        if yuzde > 0:
            kazancli_islem += 1
            toplam_kar += yuzde
        else:
            zararli_islem += 1
            toplam_zarar += yuzde

print("-" * 40)
print(f"TOPLAM İŞLEM SAYISI : {toplam_islem}")
print(f"✅ KAZANÇLI İŞLEMLER : {kazancli_islem}")
print(f"❌ ZARARLI İŞLEMLER  : {zararli_islem}")

if toplam_islem > 0:
    basari_orani = (kazancli_islem / toplam_islem) * 100
    print(f"🎯 BAŞARI ORANI      : %{basari_orani:.2f}")
else:
    print("Hiç işlem yapılmamış.")

print("=" * 40)