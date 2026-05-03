# Zemberek Python NLP API 🇹🇷🧠

Bu proje, yabancılara Türkçe öğretmek amacıyla geliştirilmiş yapay zeka destekli web platformunun "Doğal Dil İşleme (NLP)" motorudur. Java tabanlı meşhur [Zemberek](https://github.com/ahmetaa/zemberek-nlp) kütüphanesini, JPype aracılığıyla Python ortamına sararak (wrapper) dışarıya bir REST API hizmeti sunar.

## 👨‍💻 Geliştirici
- **Geliştirici:** [Arş. Gör. Kerim Gedik](https://github.com/kerimgedik07) - Tüm Zemberek kütüphanesinin Python'a entegre edilmesi, Flask API'nin oluşturulması, dilbilgisel ve morfolojik hataların Türkçe dil kurallarına (ünsüz yumuşaması vb.) uygun şekilde çözümlenmesi.
- **Web Arayüzü & Entegrasyon:** Yazılım Mühendisi Kadir Gedik tarafından hazırlanan web SaaS platformunda (https://kerimgedik.tech) tüketilir.

## 🚀 Özellikler

Bu API, Türkçe metinler üzerinde şu temel analizleri Türkçe diline uygun ve sadeleştirilmiş çıktılarla (JSON) sunar:

- **Yazım Denetimi (Spell Checker):** Hatalı yazılmış kelimeleri tespit eder ve en uygun doğru kelime önerilerini sunar.
- **Gövde Bulma (Stemming & Lemmatization):** Kelimenin eklerini atarak ana gövdesini bulur. Türkçe ses olaylarını (örn. "kitabı" -> "kitap" - ünsüz yumuşaması) dikkate alarak çalışır.
- **Morfolojik Analiz (Morphological Analysis):** Kelimenin türünü (İsim, Fiil, Sıfat vb.) ve aldığı ekleri (Zaman, Şahıs vb.) analiz eder.
- **Cümle Ögeleri (Sentence Segmentation):** Paragrafları cümlelere, cümleleri ise tokenlara (kelimelere) ayırır.
- **Heceleme:** Kelimeleri hecelerine ayırır.

## 🛠 Teknoloji Yığını
- **Programlama Dili:** Python 3.x
- **Framework:** Flask & Flask-CORS
- **NLP Motoru:** Zemberek-NLP (Java)
- **Köprü Kütüphanesi:** JPype1 (Java sanal makinesi ile Python'u haberleştirmek için)

## ⚙️ Kurulum ve Çalıştırma

1. Repoyu klonlayın:
   \`\`\`bash
   git clone https://github.com/kerimgedik07/zemberek-nlp-master.git
   cd zemberek-nlp-master
   \`\`\`

2. Gerekli kütüphaneleri yükleyin:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
   *(Not: Sisteminizde Java Development Kit - JDK yüklü olması zorunludur.)*

3. Flask sunucusunu başlatın:
   \`\`\`bash
   python app.py
   \`\`\`

API varsayılan olarak \`http://127.0.0.1:8080\` adresinde çalışmaya başlayacaktır.

## 📡 API Endpoint Örnekleri

- **\`POST /api/spell-check\`**
  Girilen metindeki yazım hatalarını ve önerileri döner.
- **\`POST /api/stem\`**
  Girilen metindeki tüm kelimelerin gövde hallerini bulur.
- **\`POST /api/morphology\`**
  Kelimelerin türsel analizlerini yapar.

## 🤝 Katkıda Bulunma
Bu proje akademik araştırma ve dil öğretimini desteklemek amacıyla geliştirilmiştir. Türkçe Doğal Dil İşleme alanına ilgi duyan herkesin katkılarına (Pull Request) açıktır.
