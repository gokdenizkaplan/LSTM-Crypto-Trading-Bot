import time
import requests
import json
import os
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime

# --- ⚙️ AYARLAR ---
SEMBOL = "BTC-USD"
TELEGRAM_TOKEN = "8586050848:AAGv4feVHuHziD1AkgwL4ELYAUCk2F1UImc"  # BotFather'dan aldığın Token
TELEGRAM_CHAT_ID = "6401973362"  # UserInfoBot'tan aldığın ID
DURUM_DOSYASI = "bot_durumu.json"

# Strateji Ayarları (Final Kararlarımız)
ALIM_ESIGI = 0.45
ENTRY_MA_LEN = 20
EXIT_MA_LEN = 100
FEATURE_LIST = ['Log_Ret', 'MFI_14', 'NATR_14', 'RSI_14', 'Dist_EMA', 'ROC_10']
LOOK_BACK_DAYS = 30

print("--- 🚀 CANLI BTC BOTU BAŞLATILIYOR ---")

# 1. MODEL VE SCALER YÜKLEME
try:
    model = load_model("sampiyon_model.h5")
    scaler = joblib.load("sampiyon_scaler.gz")
    print("✅ Yapay Zeka Yüklendi.")
except:
    print("❌ Model dosyaları bulunamadı! Lütfen aynı klasörde olduklarından emin olun.")
    exit()


# 2. TELEGRAM FONKSİYONU
def telegram_gonder(mesaj):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": mesaj, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram Hatası: {e}")


# 3. DURUM YÖNETİMİ (Hafıza)
def durumu_yukle():
    if os.path.exists(DURUM_DOSYASI):
        with open(DURUM_DOSYASI, 'r') as f:
            return json.load(f)
    return {"pozisyon": "NAKIT", "son_islem_fiyati": 0}


def durumu_kaydet(durum):
    with open(DURUM_DOSYASI, 'w') as f:
        json.dump(durum, f)


# 4. ANALİZ MOTORU
def analizi_calistir():
    durum = durumu_yukle()
    print(f"\n[{datetime.now().strftime('%H:%M')}] Veriler kontrol ediliyor...")

    # Veri Çek (Son 200 gün yeterli)
    try:
        df = yf.download(SEMBOL, period="200d", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    except:
        print("İnternet/Veri hatası. Bekleniyor...")
        return

    # İndikatörleri Hesapla
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MFI_14'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    df['NATR_14'] = ta.natr(df['High'], df['Low'], df['Close'], length=14)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df['EMA50'] = ta.ema(df['Close'], length=50)
    df['Dist_EMA'] = (df['Close'] - df['EMA50']) / df['EMA50']
    df['ROC_10'] = ta.roc(df['Close'], length=10)

    # Strateji
    df['MA_ENTRY'] = ta.ema(df['Close'], length=ENTRY_MA_LEN)
    df['MA_EXIT'] = ta.ema(df['Close'], length=EXIT_MA_LEN)

    df.dropna(inplace=True)

    # Son Veriler
    last_row = df.iloc[-1]
    guncel_fiyat = last_row['Close']
    tarih = last_row.name.strftime('%Y-%m-%d')

    # Model Tahmini
    input_df = df[FEATURE_LIST].tail(LOOK_BACK_DAYS)
    if len(input_df) < LOOK_BACK_DAYS: return  # Yeterli veri yoksa geç

    input_scaled = scaler.transform(input_df.values)
    X_pred = input_scaled.reshape(1, LOOK_BACK_DAYS, len(FEATURE_LIST))
    prob = model.predict(X_pred, verbose=0)[0][0]

    # --- KARAR MEKANİZMASI ---
    sinyal = "BEKLE"
    neden = ""

    trend_giris = guncel_fiyat > last_row['MA_ENTRY']
    trend_cikis = guncel_fiyat < last_row['MA_EXIT']
    model_onay = prob > ALIM_ESIGI

    # ALIM MANTIĞI
    if durum["pozisyon"] == "NAKIT":
        if trend_giris and model_onay:
            sinyal = "AL"
            neden = f"Trend + AI Onayı (Güven: %{prob * 100:.1f})"

    # SATIM MANTIĞI
    elif durum["pozisyon"] == "MALDA":
        if trend_cikis:
            sinyal = "SAT"
            neden = "Trend Kırıldı (EMA 100 Altı)"
        elif prob < 0.20:
            sinyal = "SAT"
            neden = "AI Çöküş Sinyali (< 0.20)"

    # --- EYLEM ---
    if sinyal == "AL":
        msg = f"🟢 **AL SİNYALİ** 🟢\n\nFiyat: ${guncel_fiyat:,.0f}\nSebep: {neden}\nModel Puanı: {prob:.2f}"
        telegram_gonder(msg)
        durum["pozisyon"] = "MALDA"
        durum["son_islem_fiyati"] = guncel_fiyat
        durumu_kaydet(durum)
        print(">>> AL SİNYALİ GÖNDERİLDİ!")

    elif sinyal == "SAT":
        kar_zarar = (guncel_fiyat - durum['son_islem_fiyati']) / durum['son_islem_fiyati']
        msg = f"🔴 **SAT SİNYALİ** 🔴\n\nFiyat: ${guncel_fiyat:,.0f}\nSebep: {neden}\nPNL: %{kar_zarar * 100:.2f}"
        telegram_gonder(msg)
        durum["pozisyon"] = "NAKIT"
        durum["son_islem_fiyati"] = guncel_fiyat
        durumu_kaydet(durum)
        print(">>> SAT SİNYALİ GÖNDERİLDİ!")

    else:
        print(f"Sinyal Yok. Pozisyon: {durum['pozisyon']} | Fiyat: {guncel_fiyat:.0f} | AI: {prob:.2f}")


# --- 5. SONSUZ DÖNGÜ ---
telegram_gonder(f"🤖 **Bot Başlatıldı!**\nStrateji: Geniş Bant\nHedef: {SEMBOL}")

while True:
    try:
        analizi_calistir()
    except Exception as e:
        print(f"Hata oluştu: {e}")
        time.sleep(10)  # Hata olursa 10sn bekle

    # Her 1 saatte bir kontrol et (3600 saniye)
    # Daha sık kontrol istersen süreyi düşür (örn: 900 = 15dk)
    print("Beklemeye geçildi (1 Saat)...")
    time.sleep(36)