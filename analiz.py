import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler

# --- 🎯 ANALİZ AYARLARI ---
SEMBOL = "BTC-USD"
MODEL_DOSYASI = "sampiyon_model.h5"
SCALER_DOSYASI = "sampiyon_scaler.gz"

# HODL+ Stratejisi Eşikleri (Analizde referans olması için)
BOGA_GIRIS_ESIGI = 0.35
AYI_GIRIS_ESIGI = 0.60

# STANDART ANALİZ EŞİĞİ (Raporlar için orta nokta)
GENEL_ESIK = 0.50

# EĞİTİM İLE AYNI OLMAK ZORUNDA
FEATURE_LIST = ['Log_Ret', 'MFI_14', 'NATR_14', 'RSI_14', 'Dist_EMA', 'ROC_10']
LOOK_BACK_DAYS = 30
FUTURE_DAYS = 3      # Eğitimde 3 gün sonrasını hedeflemiştik
THRESHOLD = 0.02     # %2 Kar hedefi

# 1. YÜKLEME
print(f"--- 🔍 MODEL ANALİZİ: {SEMBOL} ---")
try:
    model = load_model(MODEL_DOSYASI)
    scaler = joblib.load(SCALER_DOSYASI)
    print("✅ Model ve Scaler başarıyla yüklendi.")
except Exception as e:
    print(f"❌ HATA: {e}")
    print("Lütfen 'sampiyon_model.h5' ve 'sampiyon_scaler.gz' dosyalarının klasörde olduğundan emin olun.")
    exit()

# 2. VERİ (Son 3 Yıl - Hem Ayı Hem Boğa görmek için)
print("Veri çekiliyor (Son 3 Yıl)...")
df = yf.download(SEMBOL, period="3y", interval="1d", progress=False)

# 🛠️ MultiIndex Düzeltmesi (Eğitimdeki gibi)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 3. İNDİKATÖRLER (Eğitim Formülleriyle BİREBİR AYNI)
df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
df['MFI_14'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
df['NATR_14'] = ta.natr(df['High'], df['Low'], df['Close'], length=14)
df['RSI_14'] = ta.rsi(df['Close'], length=14)
df['EMA50'] = ta.ema(df['Close'], length=50)
df['Dist_EMA'] = (df['Close'] - df['EMA50']) / df['EMA50']
df['ROC_10'] = ta.roc(df['Close'], length=10)

# 4. GERÇEK HEDEF (Ground Truth)
# Eğitimde ne öğrettiysek burada da aynısını test ediyoruz
df['Future_Close'] = df['Close'].shift(-FUTURE_DAYS)
df['Change'] = (df['Future_Close'] - df['Close']) / df['Close']
df['Target'] = (df['Change'] > THRESHOLD).astype(int) # 1: Yükseliş, 0: Bekle

df.dropna(inplace=True)

# 5. TAHMİN ÜRETME
# Sadece transform yapıyoruz, fit yok!
input_data = scaler.transform(df[FEATURE_LIST].values)

X, y_true = [], []
for i in range(LOOK_BACK_DAYS, len(input_data) - FUTURE_DAYS):
    X.append(input_data[i-LOOK_BACK_DAYS:i])
    y_true.append(df['Target'].iloc[i])

X = np.array(X)
y_true = np.array(y_true)

print("Tahminler üretiliyor...")
probs = model.predict(X, verbose=0)
y_pred_proba = probs.flatten()
y_pred_class = (y_pred_proba > GENEL_ESIK).astype(int)

# --- 📊 GRAFİK 1: KARIŞIKLIK MATRİSİ ---
cm = confusion_matrix(y_true, y_pred_class)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Düşüş/Yatay', 'Yükseliş (>%2)'],
            yticklabels=['Gerçek Düşüş', 'Gerçek Yükseliş'])
plt.xlabel('Model Tahmini')
plt.ylabel('Gerçek Durum')
plt.title(f'1. Confusion Matrix (Genel Eşik: {GENEL_ESIK})')
plt.show()

# --- 📊 GRAFİK 2: ISI HARİTASI ---
plt.figure(figsize=(10, 8))
# Sadece özellikler ve hedef arasındaki ilişki
analiz_df = df[FEATURE_LIST].copy()
analiz_df['Target'] = df['Target']
corr_df = analiz_df.corr()
sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('2. Özellikler ve Hedef İlişkisi')
plt.show()

# --- 📊 GRAFİK 3: PUAN DAĞILIMI VE STRATEJİ EŞİKLERİ ---
plt.figure(figsize=(10, 6))
plt.hist(y_pred_proba, bins=50, color='#673ab7', alpha=0.7, edgecolor='black', label='Model Puanları')

# HODL+ Strateji Çizgileri
plt.axvline(BOGA_GIRIS_ESIGI, color='green', linestyle='dashed', linewidth=2, label=f'Boğa Giriş ({BOGA_GIRIS_ESIGI})')
plt.axvline(AYI_GIRIS_ESIGI, color='red', linestyle='dashed', linewidth=2, label=f'Ayı Giriş ({AYI_GIRIS_ESIGI})')

plt.title('3. Model Güven Puanı Dağılımı ve HODL+ Eşikleri')
plt.xlabel('0 (Kesin Düşüş) <---> 1 (Kesin Yükseliş)')
plt.ylabel('Gün Sayısı')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# --- 📊 GRAFİK 4: ROC EĞRİSİ ---
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('4. ROC Eğrisi (Ayırt Etme Gücü)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# --- 📝 RAPOR ---
print("\n" + "="*50)
print(f"SINIFLANDIRMA RAPORU (Genel Başarı)")
print("-" * 50)
print(classification_report(y_true, y_pred_class, target_names=['BEKLE', 'YÜKSELİŞ']))
print("-" * 50)
print("YORUM:")
print("Eğer 'Precision' (Keskinlik) yüksekse: Modelin 'AL' dediği genelde tutuyor demektir.")
print("Eğer 'Recall' (Duyarlılık) yüksekse: Model fırsatları kaçırmıyor demektir.")
print("="*50)