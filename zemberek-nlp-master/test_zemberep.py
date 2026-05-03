import time
import logging

from zemberek import (
    TurkishSpellChecker,
    TurkishSentenceNormalizer,
    TurkishSentenceExtractor,
    TurkishMorphology,
    TurkishTokenizer
)

logger = logging.getLogger(__name__)

examples = ["Yrn okua gidicem",
            "Tmm, yarin havuza giricem ve aksama kadar yaticam :)",
            "ah aynen ya annemde fark ettı siz evinizden cıkmayın diyo",
            "gercek mı bu? Yuh! Artık unutulması bile beklenmiyo",
            "Hayır hayat telaşm olmasa alacam buraları gökdelen dikicem.",
            "yok hocam kesınlıkle oyle birşey yok",
            "herseyi soyle hayatında olmaması gerek bence boyle ınsanların falan baskı yapıyosa",
            "email adresim zemberek_python@loodos.com",
            "Kredi başvrusu yapmk istiyrum.",
            "Bankanizin hesp blgilerini ogrenmek istyorum."]

morphology = TurkishMorphology.create_with_defaults()

# SENTENCE NORMALIZATION
print("=" * 60)
print("CÜMLE NORMALİZASYONU")
print("=" * 60)
start = time.time()
normalizer = TurkishSentenceNormalizer(morphology)
print(f"Normalizer oluşturuldu: {time.time() - start:.2f} s\n")

start = time.time()
for example in examples:
    print(f"Girdi  : {example}")
    print(f"Çıktı  : {normalizer.normalize(example)}\n")
print(f"Toplam normalizasyon süresi: {time.time() - start:.2f} s\n")

# SPELLING SUGGESTION
print("=" * 60)
print("YAZIM ÖNERİLERİ")
print("=" * 60)
start = time.time()
sc = TurkishSpellChecker(morphology)
print(f"Spell checker oluşturuldu: {time.time() - start:.2f} s\n")

li = ["okuyablirim", "tartısıyor", "Ankar'ada", "knlıca", "yapablrim", "kıredi", "geldm", "geliyom", "aldm", "asln"]
start = time.time()
for word in li:
    print(f"{word} => {' '.join(sc.suggest_for_word(word))}")
print(f"\nYazım kontrolü süresi: {time.time() - start:.2f} s\n")

# SENTENCE BOUNDARY DETECTION
print("=" * 60)
print("CÜMLE SINIRI TESPİTİ")
print("=" * 60)
start = time.time()
extractor = TurkishSentenceExtractor()
print(f"Extractor oluşturuldu: {time.time() - start:.4f} s\n")

text = ("İnsanoğlu aslında ne para ne sevgi ne kariyer ne şöhret ne de çevre ile sonsuza dek mutlu olabilecek bir "
        "yapıya sahiptir. Dış kaynaklardan gelebilecek bu mutluluklar sadece belirli bir zaman için insanı mutlu "
        "kılıyor. Kişi bu kaynakları elde ettiği zaman belirli bir dönem için kendini iyi hissediyor, ancak alışma "
        "dönemine girdiği andan itibaren bu iyilik hali hızla tükeniyor. Mutlu olma sanatının özü bu değildir. Gerçek "
        "mutluluk, kişinin her türlü olaya ve duruma karşı kendini pozitif tutarak mutlu hissedebilmesi halidir. Bu "
        "davranış şeklini edinen insan, zor günlerde güçlü, mutlu günlerde zevk alan biri olur ve mutluluğu kalıcı "
        "kılar. ")

start = time.time()
sentences = extractor.from_paragraph(text)
print(f"Cümleler ayrıldı: {time.time() - start:.4f} s\n")

for i, sentence in enumerate(sentences, 1):
    print(f"  {i}. {sentence}")
print()

# SINGLE WORD MORPHOLOGICAL ANALYSIS
print("=" * 60)
print("TEK KELİME MORFOLOJİK ANALİZ")
print("=" * 60)
results = morphology.analyze("kalemin")
print(f"'kalemin' analizi:")
for result in results:
    print(f"  {result}")
print()

# SENTENCE ANALYSIS AND DISAMBIGUATION
print("=" * 60)
print("CÜMLE ANALİZİ VE ANLAM AYRIŞTIRMA")
print("=" * 60)

sentence = "Yarın kar yağacak."
analysis = morphology.analyze_sentence(sentence)
after = morphology.disambiguate(sentence, analysis)

print("\nAyrıştırma öncesi:")
for e in analysis:
    print(f"  Kelime = {e.inp}")
    for s in e:
        print(f"    {s.format_string()}")

print("\nAyrıştırma sonrası:")
for s in after.best_analysis():
    print(f"  {s.format_string()}")
print()

# TOKENIZATION
print("=" * 60)
print("TOKENİZASYON")
print("=" * 60)
tokenizer = TurkishTokenizer.DEFAULT

tokens = tokenizer.tokenize("Saat 12:00.")
for token in tokens:
    print(f"  İçerik = {token.content}")
    print(f"  Tip    = {token.type_.name}")
    print(f"  Başla  = {token.start}")
    print(f"  Bitir  = {token.end}\n")

print("=" * 60)
print("Tüm testler başarıyla tamamlandı!")
print("=" * 60)
