import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


SEMBOL = "BTC-USD"
MODEL_DOSYASI = "sampiyon_model.h5"
SCALER_DOSYASI = "sampiyon_scaler.gz"


ALIM_ESIGI = 0.45

# STRATEJİ EMA'LARI
ENTRY_MA_LEN = 20  # EMA 20 (Hızlı Giriş)
EXIT_MA_LEN = 100  # EMA 100 (Güvenli Çıkış)

KOMISYON = 0.001
BASLANGIC_BAKIYE = 1000


FEATURE_LIST = ['Log_Ret', 'MFI_14', 'NATR_14', 'RSI_14', 'Dist_EMA', 'ROC_10']
LOOK_BACK_DAYS = 30
BUGUN = datetime.now().strftime('%Y-%m-%d')

print(f"--- FİNAL BACKTEST: GENİŞ BANT (Eşik {ALIM_ESIGI}) ---")

# 1. Yükleme
try:
    model = load_model(MODEL_DOSYASI)
    scaler = MinMaxScaler(feature_range=(0, 1))
    print(">>> Model Yüklendi.")
except Exception as e:
    print(f"HATA: {e}")
    exit()

# 2. Veri
print("Veri indiriliyor...")
df = yf.download(SEMBOL, start="2019-06-01", end=BUGUN, interval="1d", progress=False)
if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

# 3. İndikatörler (Tüm 6 Özellik)
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
df = df[df.index >= '2020-01-01']

# Scaling
scaled_data = scaler.fit_transform(df[FEATURE_LIST].values)

# 4. Motor
usdt_balance = BASLANGIC_BAKIYE
btc_balance = 0
in_position = False
entry_price = 0

portfolio_history = []
trade_log = []

for i in range(LOOK_BACK_DAYS, len(df) - 1):
    current_price = df['Close'].iloc[i]
    ma_entry = df['MA_ENTRY'].iloc[i]
    ma_exit = df['MA_EXIT'].iloc[i]
    date = df.index[i]

    val = usdt_balance + (btc_balance * current_price)
    portfolio_history.append(val)

    X_window = scaled_data[i - LOOK_BACK_DAYS:i].reshape(1, LOOK_BACK_DAYS, len(FEATURE_LIST))
    probs = model.predict(X_window, verbose=0)
    prob_buy = probs[0][0]

    # --- STRATEJİ ---
    if in_position:
        should_sell = False

        # ÇIKIŞ: EMA 100 Kırılınca SAT
        if current_price < ma_exit:
            should_sell = True
            reason = f"Ana Trend Bitti (EMA {EXIT_MA_LEN})"

        # Model ACİL ÇIKIŞ (Çok düşük güven)
        elif prob_buy < 0.20:
            should_sell = True
            reason = "Model Çöküş Sinyali"

        if should_sell:
            usdt_balance = (btc_balance * current_price) * (1 - KOMISYON)
            btc_balance = 0
            pnl = (current_price - entry_price) / entry_price
            trade_log.append(f"SAT ({reason}): {date.date()} | PNL: %{pnl * 100:.2f}")
            in_position = False

    else:
        # GİRİŞ KOŞULU (Eşik 0.45 ile AL)
        model_onay = prob_buy > ALIM_ESIGI
        trend_onay = current_price > ma_entry

        if model_onay and trend_onay:
            btc_balance = (usdt_balance / current_price) * (1 - KOMISYON)
            usdt_balance = 0
            entry_price = current_price

            trade_log.append(f"AL  (V4): {date.date()} | Fiyat: {entry_price:.0f} | Güven: %{prob_buy * 100:.1f}")
            in_position = True

# Sonuçlar
last_val = usdt_balance + (btc_balance * df['Close'].iloc[-1])
portfolio_history.append(last_val)

dates = df.index[LOOK_BACK_DAYS:LOOK_BACK_DAYS + len(portfolio_history)]
perf_df = pd.DataFrame({'Bot_Balance': portfolio_history,
                        'BTC_Price': df['Close'].iloc[LOOK_BACK_DAYS:LOOK_BACK_DAYS + len(portfolio_history)].values},
                       index=dates)

start_date = dates[0] - pd.Timedelta(days=1)
start_row = pd.DataFrame({'Bot_Balance': [BASLANGIC_BAKIYE], 'BTC_Price': [df['Close'].iloc[LOOK_BACK_DAYS]]},
                         index=[start_date])
perf_df = pd.concat([start_row, perf_df])

annual_returns = perf_df.resample('YE').last().pct_change() * 100
annual_returns.dropna(inplace=True)

print("\n" + "=" * 50)
print(f"{'YIL':<6} | {'BOT GETİRİSİ':<15} | {'HODL GETİRİSİ':<15} | {'DURUM'}")
print("-" * 50)

for date, row in annual_returns.iterrows():
    year = date.year
    bot_ret = row['Bot_Balance']
    hodl_ret = row['BTC_Price']
    status = "✅ YENDİ" if bot_ret > hodl_ret else "❌ GERİDE"
    print(f"{year:<6} | % {bot_ret:<13.2f} | % {hodl_ret:<13.2f} | {status}")

total_ret = ((portfolio_history[-1] - BASLANGIC_BAKIYE) / BASLANGIC_BAKIYE) * 100
hodl_total = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100

print("-" * 50)
print(f"TOPLAM | % {total_ret:<13.2f} | % {hodl_total:<13.2f} | {'👑 ŞAMPİYON' if total_ret > hodl_total else ''}")
print("=" * 50)

plt.figure(figsize=(12, 6))
plt.yscale('log')
plt.plot(dates, perf_df['BTC_Price'][1:] / perf_df['BTC_Price'].iloc[0] * BASLANGIC_BAKIYE, label='HODL', color='gray',
         alpha=0.5)
plt.plot(dates, portfolio_history, label=f'Final Bot (EMA {EXIT_MA_LEN} Çıkış)', color='green', linewidth=2)
plt.title('Final Strateji: Geniş Bant (HODL Katili)')
plt.legend()
plt.xlim(dates[0], dates[-1])
plt.show()
