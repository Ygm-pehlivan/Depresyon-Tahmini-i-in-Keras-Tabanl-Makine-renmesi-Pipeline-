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
