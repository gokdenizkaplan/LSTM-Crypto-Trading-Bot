import ccxt
import time
import json
import os
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime

# --- 🔐 OKX VE STRATEJİ AYARLARI ---
API_KEY = ""
API_SECRET = ""
API_PASSWORD = ""

SEMBOL = "BTC/USDT"  # OKX sembol formatı
YF_SEMBOL = "BTC-USD"  # Yahoo Finance formatı (Veri çekmek için)

# Strateji (Değiştirme, modelinle aynı kalsın)
ALIM_ESIGI = 0.45
ENTRY_MA_LEN = 20
EXIT_MA_LEN = 100
FEATURE_LIST = ['Log_Ret', 'MFI_14', 'NATR_14', 'RSI_14', 'Dist_EMA', 'ROC_10']
LOOK_BACK_DAYS = 30

# Durum Dosyası (Botun hafızası)
DURUM_DOSYASI = "bot_durumu.json"

print("--- 🤖 OTOMATİK OKX BOTU BAŞLATILIYOR ---")

# 1. OKX BAĞLANTISI
try:
    exchange = ccxt.okx({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'password': API_PASSWORD,
        'enableRateLimit': True,
    })
    # Bağlantı testi
    bakiye = exchange.fetch_balance()
    print(f"✅ OKX Bağlantısı Başarılı! USDT Bakiyesi: {bakiye['USDT']['free']:.2f}$")
except Exception as e:
    print(f"❌ OKX Bağlantı Hatası: {e}")
    exit()

# 2. MODEL YÜKLEME
try:
    model = load_model("sampiyon_model.h5")
    scaler = joblib.load("sampiyon_scaler.gz")
    print("✅ Yapay Zeka Yüklendi.")
except:
    print("❌ Model dosyaları eksik! (sampiyon_model.h5 ve scaler)")
    exit()


# 3. YARDIMCI FONKSİYONLAR
def durumu_yukle():
    if os.path.exists(DURUM_DOSYASI):
        with open(DURUM_DOSYASI, 'r') as f:
            return json.load(f)
    # Varsayılan durum
    return {"pozisyon": "NAKIT"}


def durumu_kaydet(durum):
    with open(DURUM_DOSYASI, 'w') as f:
        json.dump(durum, f)


def emir_ver(islem_tipi, miktar=None):
    """
    islem_tipi: 'buy' veya 'sell'
    miktar: BTC cinsinden miktar (None ise tüm bakiyeyi kullanır)
    """
    try:
        # Piyasadan anlık fiyat al
        ticker = exchange.fetch_ticker(SEMBOL)
        fiyat = ticker['last']

        if islem_tipi == 'buy':
            # Mevcut USDT miktarını al
            usdt_bakiye = exchange.fetch_balance()['USDT']['free']
            # Güvenlik: Bakiyenin %99'unu kullan (Komisyon için pay bırak)
            alinacak_miktar = (usdt_bakiye * 0.99) / fiyat

            # OKX Market Emri
            order = exchange.create_market_buy_order(SEMBOL, alinacak_miktar)
            print(f"🟢 ALIM EMRİ GİRİLDİ: {alinacak_miktar:.6f} BTC @ {fiyat}")
            return True

        elif islem_tipi == 'sell':
            # Mevcut BTC miktarını al
            btc_bakiye = exchange.fetch_balance()['BTC']['free']

            # OKX Market Emri
            order = exchange.create_market_sell_order(SEMBOL, btc_bakiye)
            print(f"🔴 SATIM EMRİ GİRİLDİ: {btc_bakiye:.6f} BTC @ {fiyat}")
            return True

    except Exception as e:
        print(f"🚨 EMİR HATASI: {e}")
        return False


# 4. ANALİZ VE İŞLEM MOTORU
def analizi_calistir():
    durum = durumu_yukle()
    print(f"\n[{datetime.now().strftime('%H:%M')}] Piyasa Analiz Ediliyor...")

    # Veri Çekme (Analiz için Yahoo kullanıyoruz, işlem için OKX)
    try:
        df = yf.download(YF_SEMBOL, period="200d", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    except:
        print("Veri hatası. Bekleniyor...")
        return

    # İndikatörler (Modelin Gözleri)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MFI_14'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    df['NATR_14'] = ta.natr(df['High'], df['Low'], df['Close'], length=14)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df['EMA50'] = ta.ema(df['Close'], length=50)
    df['Dist_EMA'] = (df['Close'] - df['EMA50']) / df['EMA50']
    df['ROC_10'] = ta.roc(df['Close'], length=10)

    # Strateji EMA'ları
    df['MA_ENTRY'] = ta.ema(df['Close'], length=ENTRY_MA_LEN)
    df['MA_EXIT'] = ta.ema(df['Close'], length=EXIT_MA_LEN)

    df.dropna(inplace=True)
    last_row = df.iloc[-1]
    guncel_fiyat = last_row['Close']

    # Model Tahmini
    input_df = df[FEATURE_LIST].tail(LOOK_BACK_DAYS)
    input_scaled = scaler.transform(input_df.values)
    X_pred = input_scaled.reshape(1, LOOK_BACK_DAYS, len(FEATURE_LIST))
    prob = model.predict(X_pred, verbose=0)[0][0]

    # --- KARAR MEKANİZMASI ---
    sinyal = "BEKLE"
    trend_giris = guncel_fiyat > last_row['MA_ENTRY']
    trend_cikis = guncel_fiyat < last_row['MA_EXIT']
    model_onay = prob > ALIM_ESIGI

    print(
        f"Fiyat: {guncel_fiyat:.0f} | EMA20: {last_row['MA_ENTRY']:.0f} | AI Puanı: {prob:.2f} | Pozisyon: {durum['pozisyon']}")

    # AL SİNYALİ
    if durum["pozisyon"] == "NAKIT":
        if trend_giris and model_onay:
            print(">>> KOŞULLAR SAĞLANDI! ALIM YAPILIYOR...")
            basarili = emir_ver('buy')
            if basarili:
                durum["pozisyon"] = "MALDA"
                durumu_kaydet(durum)

    # SAT SİNYALİ
    elif durum["pozisyon"] == "MALDA":
        satis_sebebi = ""
        if trend_cikis:
            satis_sebebi = "Trend Kırıldı"
        elif prob < 0.20:
            satis_sebebi = "Model Çöküş Sinyali"

        if satis_sebebi:
            print(f">>> {satis_sebebi}! SATIŞ YAPILIYOR...")
            basarili = emir_ver('sell')
            if basarili:
                durum["pozisyon"] = "NAKIT"
                durumu_kaydet(durum)


# 5. DÖNGÜ
print("Bot aktif. Her 1 saatte bir kontrol edecek.")
while True:
    try:
        analizi_calistir()
    except Exception as e:
        print(f"Genel Hata: {e}")

    time.sleep(3)  # 1 Saat bekle

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