# Depresyon-Tahmini-i-in-Keras-Tabanl-Makine-renmesi-Pipeline-
Bu proje, bireylerin depresyon durumunu tahmin etmek için Python, scikit-learn ve Keras ile geliştirilen bir makine öğrenmesi pipeline’ıdır. Veri temizleme, ön işleme, model eğitimi ve görselleştirme adımlarıyla %93.38 doğruluk elde edilmiş; sınıf bazlı metrikler ve confusion matrix ile değerlendirilmiştir.
# Depresyon Tahmini Makine Öğrenmesi Projesi

## 🔍 Amaç
Bu proje, bireylerin depresyon durumunu çeşitli demografik ve psikolojik özelliklere göre tahmin etmeyi amaçlamaktadır.

## 🧠 Kullanılan Teknolojiler
- Python (pandas, numpy)
- scikit-learn (Pipeline, ColumnTransformer, OneHotEncoder, StandardScaler)
- Keras (Sequential model)
- Görselleştirme: Matplotlib, seaborn

## 📊 Veri Seti
- `train.csv` dosyası, bireylerin yaş, cinsiyet, meslek, uyku süresi, akademik/iş baskısı gibi özelliklerini içermektedir.
- Hedef değişken: `Depression` (0: Yok, 1: Var)

## ⚙️ Pipeline Adımları
1. Veri temizleme (`UNKNOWN` değerlerin işlenmesi)
2. Kategorik ve sayısal sütun ayrımı
3. OneHotEncoding + StandardScaler ile ön işleme
4. Eğitim/doğrulama ayrımı (`train_test_split`)
5. Keras ile sinir ağı modeli kurulumu
6. Model eğitimi ve doğrulama
7. Başarı metrikleri ve görselleştirme

## 📈 Sonuçlar
- Doğruluk: **%93.38**[Depresyon Tahmini Modeli-2.pdf](https://github.com/user-attachments/files/23173730/Depresyon.Tahmini.Modeli-2.pdf)[Depresyon Tahmini Modeli-2.pdf](https://github.com/user-attachments/files/23173732/Depresyon.Tahmini.Modeli-2.pdf)


- Sınıf 0 için F1-Score: **0.96**
- Sınıf 1 için F1-Score: **0.81**
- Confusion Matrix ve başarı metrikleri görselleştirilmiştir.

## 🗺️ Veri Akış Diyagramı
Proje adımları, görsel bir diyagram ile sunulmuştur.


## 👩‍💻 Hazırlayan
Yağmur Pehlivan


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Veri yükleme
train = pd.read_csv('/Users/yagmurphlvn/Downloads/train.csv')

# Gereksiz sütunları çıkar
X = train.drop(columns=['Depression', 'Name'])
y = train['Depression']

# Sayısal sütunlar
numeric_cols = ['Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction',
                'Job Satisfaction', 'Financial Stress', 'Work/Study Hours', 'Age']

# Kategorik sütunlar
categorical_cols = ['Gender', 'City', 'Working Professional or Student', 'Profession',
                    'Sleep Duration', 'Dietary Habits', 'Degree',
                    'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

# 1. 'UNKNOWN' gibi metinleri sayısal sütunlardan temizle
for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')  # 'UNKNOWN' → NaN
    X[col].fillna(X[col].mean(), inplace=True)

# 2. Kategorik sütunlardaki 'UNKNOWN' değerleri mod ile doldur
for col in categorical_cols:
    X[col] = X[col].replace('UNKNOWN', np.nan)
    X[col].fillna(X[col].mode()[0], inplace=True)

# 3. Ön işleme pipeline'ı
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('prep', preprocessor),
    ('scale', StandardScaler())
])

# 4. Veriyi ayır ve dönüştür
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = pipeline.fit_transform(X_train)
X_val_scaled = pipeline.transform(X_val)

# 5. Basit model
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Modeli eğit
model.fit(X_train_scaled, y_train, epochs=10, validation_data=(X_val_scaled, y_val))

# 7. Değerlendirme
loss, acc = model.evaluate(X_val_scaled, y_val)
print(f"Doğruluk: {acc * 100:.2f}%")
