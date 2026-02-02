import os
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# --- SEED SABİTLEME ---
np.random.seed(42)
tf.random.set_seed(42)

print("--- 🧠 MODEL EĞİTİMİ: DENGELİ VE SADELEŞTİRİLMİŞ MOD ---")

# --- 1. AYARLAR ---
SEMBOL = "BTC-USD"
START_DATE = "2017-01-01"
LOOK_BACK = 30
TARGET_DAYS = 3
TARGET_THRESHOLD = 0.015  # %1.5 hedef (İdeal)

# --- 2. VERİ ---
print(f"1. Veri indiriliyor: {SEMBOL}...")
df = yf.download(SEMBOL, start=START_DATE, interval="1d", progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df.dropna(inplace=True)

# --- 3. ÖZELLİKLER ---
df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
df['MFI_14'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
df['NATR_14'] = ta.natr(df['High'], df['Low'], df['Close'], length=14)
df['RSI_14'] = ta.rsi(df['Close'], length=14)
df['EMA50'] = ta.ema(df['Close'], length=50)
df['Dist_EMA'] = (df['Close'] - df['EMA50']) / df['EMA50']
df['ROC_10'] = ta.roc(df['Close'], length=10)

# HEDEF
df['Future_Close'] = df['Close'].shift(-TARGET_DAYS)
df['Change'] = (df['Future_Close'] - df['Close']) / df['Close']
df['Target'] = (df['Change'] > TARGET_THRESHOLD).astype(int)

df.dropna(inplace=True)

FEATURE_LIST = ['Log_Ret', 'MFI_14', 'NATR_14', 'RSI_14', 'Dist_EMA', 'ROC_10']

# --- 4. VERİ HAZIRLIĞI ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[FEATURE_LIST].values)
target_data = df['Target'].values

X, y = [], []
for i in range(LOOK_BACK, len(scaled_data) - TARGET_DAYS):
    X.append(scaled_data[i - LOOK_BACK:i])
    y.append(target_data[i])

X, y = np.array(X), np.array(y)

split = int(len(X) * 0.85)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- ⚖️ OTOMATİK DENGELEME (En Önemli Kısım) ---
# Manuel 1:5 yerine, verinin doğasına uygun matematiksel dengeyi buluyoruz.
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

print(f"Hesaplanan İdeal Ağırlıklar: {class_weights_dict}")

# --- 5. MODEL MİMARİSİ (SADELEŞTİRİLMİŞ) ---
# Daha az nöron = Daha az kafa karışıklığı = Daha net kararlar
model = Sequential([
    LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.3),  # Ezberlemeyi önlemek için biraz unutturuyoruz

    LSTM(units=32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),

    Dense(units=16, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.001)  # Standart hız
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# --- 6. EĞİTİM ---
checkpoint = ModelCheckpoint("sampiyon_model.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

print("3. Model Eğitiliyor... (Dengeli Mod)")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights_dict,  # Otomatik ağırlıklar
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

joblib.dump(scaler, "sampiyon_scaler.gz")
print("\n>>> DENGELİ EĞİTİM TAMAMLANDI! 🚀")