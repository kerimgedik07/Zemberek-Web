"""
Zemberek NLP Web Uygulaması
Flask backend - Türkçe doğal dil işleme araçları
"""

import re
import time
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from zemberek import (
    TurkishSpellChecker,
    TurkishSentenceNormalizer,
    TurkishSentenceExtractor,
    TurkishMorphology,
    TurkishTokenizer
)

# ─────────────────────────────────────────────
# Cümlenin Ögeleri — Söz Öbeği Tabanlı Analiz
# Referans: umutakter/Zemberek-Python-Otomata-Cumlenin-Ogeleri
# Geliştirme: Phrase-based (söz öbeği) yaklaşımı
# ─────────────────────────────────────────────

# Sıfat-fiil / Zarf-fiil etiketleri (bunlar fiil ama yüklem DEĞİL)
_PARTICIPLE_TAGS = frozenset([
    'FutPart', 'PastPart', 'PresPart', 'AorPart', 'NarrPart',
    'FeelLike', 'NotState',
])
_GERUND_TAGS = frozenset([
    'WithoutDoing', 'While', 'When', 'SinceDoingSo',
    'ByDoingSo', 'AfterDoingSo', 'AsLongAs',
])
_INFINITIVE_TAGS = frozenset(['Inf1', 'Inf2', 'Inf3'])


def _extract_word_info(raw_analysis: str, word_text: str) -> dict:
    """Zemberek ham analizinden kelime bilgisini çıkarır."""
    info = {
        'word': word_text,
        'raw': raw_analysis,
        'pos': '',
        'case': 'Nom',
        'has_poss': False,
        'poss_person': 0,
        'is_verb': False,          # Çekimli fiil (yüklem adayı)
        'is_participle': False,    # Sıfat-fiil
        'is_gerund': False,        # Zarf-fiil
        'is_infinitive': False,    # Mastar
        'is_adj': False,
        'is_det': False,
        'is_adv': False,
        'is_punc': False,
        'is_conj': False,
        'is_postp': False,
        'verb_person': 0,          # 1, 2, 3
        'verb_plural': False,
    }

    if not word_text.strip() or re.match(r'^[^\w]+$', word_text, re.UNICODE):
        info['is_punc'] = True
        return info

    # POS
    pos_m = re.search(
        r':(Noun|Verb|Adj|Adv|Pron|Num|Conj|Postp|Punc|Ques|Interj|Dup|Det)',
        raw_analysis)
    if pos_m:
        info['pos'] = pos_m.group(1)
    if info['pos'] == 'Punc':
        info['is_punc'] = True
        return info

    # Case
    case_m = re.search(r':(Nom|Acc|Gen|Dat|Loc|Abl|Ins|Equ)', raw_analysis)
    if case_m:
        info['case'] = case_m.group(1)

    # İyelik (possessive) — Zemberek formatı: "si:P3sg" veya "+P3sg"
    poss_m = re.search(r'[+:]P([123])(sg|pl)', raw_analysis)
    if poss_m:
        info['has_poss'] = True
        info['poss_person'] = int(poss_m.group(1))

    # Fiil analizi
    if info['pos'] == 'Verb':
        if any(t in raw_analysis for t in _PARTICIPLE_TAGS):
            info['is_participle'] = True
        elif any(t in raw_analysis for t in _GERUND_TAGS):
            info['is_gerund'] = True
        elif any(t in raw_analysis for t in _INFINITIVE_TAGS):
            info['is_infinitive'] = True
        else:
            info['is_verb'] = True
            # Kişi/sayı
            agr_m = re.search(r'\+A([123])(sg|pl)', raw_analysis)
            if agr_m:
                info['verb_person'] = int(agr_m.group(1))
                info['verb_plural'] = (agr_m.group(2) == 'pl')

    info['is_adj'] = (info['pos'] == 'Adj' and not info['is_participle'])
    info['is_det'] = (info['pos'] == 'Det')
    info['is_adv'] = (info['pos'] == 'Adv')
    info['is_conj'] = (info['pos'] in ('Conj',))
    info['is_postp'] = (info['pos'] in ('Postp',))
    # Özel isim: "Ege", "Ahmet" vb.
    info['is_proper'] = ('Prop' in raw_analysis)
    # Türemiş isim: -lik/-lık gibi eklerle yapılmış (Ness, Become, Rel...)
    info['has_derivation'] = bool(re.search(r':Ness|:Become|:Acquire|:With\b|:Without\b|:Rel\b|:Ly\b', raw_analysis))

    # İsim/sıfat yüklemi (copula): çirkindi, güzeldir, öğretmenmiş, öğrenciydi vb.
    # Zemberek: "Adj|di:Cop+Past+A3sg" veya "Noun+A3sg|dir:Cop+Pres+A3sg"
    info['is_copula'] = False
    if not info['is_verb'] and info['pos'] in ('Adj', 'Noun', 'Adv', 'Num', 'Pron', 'Dup'):
        info['is_copula'] = bool(
            re.search(r':Cop|Cop\+', raw_analysis) or
            re.search(r'\|[^|]*(Past|Narr|Cond|Prog|Aor)\+A[123](sg|pl)', raw_analysis)
        )

    return info


def cumle_ogelerini_bul(sentence: str, morphology_obj) -> list:
    """
    Geriye yayılma (backward propagation) ile cümle ögesi atar.

    Pass 1: Her kelimeye POS/case'e göre ilk rol ata
    Pass 2: Sağdan sola yayılma
      - Adj/Det/sıfat-fiil → baş ismin rolünü al
      - Nom isim + P3sg isim → tamlama
      - Dolaylı Tümleç / Zarf Tümleci + sıfat-fiil → baş ismin rolünü al
    Pass 3: Belirtisiz nesne tespiti
    """
    try:
        analysis = morphology_obj.analyze_sentence(sentence)
        after = morphology_obj.disambiguate(sentence, analysis)
        best = after.best_analysis()

        words = []
        for i, s in enumerate(best):
            raw = s.format_string()
            word_text = analysis[i].inp
            words.append(_extract_word_info(raw, word_text))

        n = len(words)

        # ── Pass 1: İlk rol ataması ──
        for w in words:
            if w['is_punc']:
                w['oge'] = '—'
            elif w['is_verb']:
                w['oge'] = 'Yüklem'
            elif w['is_copula']:          # isim/sıfat yüklemi (çirkindi, güzeldir)
                w['oge'] = 'Yüklem'
            elif w['is_adv'] or w['is_gerund']:
                w['oge'] = 'Zarf Tümleci'
            elif w['is_postp']:
                w['oge'] = 'Edat Tümleci'
            elif w['is_conj']:
                w['oge'] = '—'
            elif w['pos'] in ('Noun', 'Pron', 'Num') or w['is_infinitive']:
                case = w['case']
                if case == 'Acc':
                    w['oge'] = 'Belirtili Nesne'
                elif case in ('Dat', 'Loc', 'Abl'):
                    w['oge'] = 'Dolaylı Tümleç'
                elif case == 'Ins':
                    w['oge'] = 'Zarf Tümleci'  # özveriyle, sevinçle vb. → nasıl?
                elif case == 'Equ':
                    w['oge'] = 'Zarf Tümleci'
                else:
                    w['oge'] = 'Özne'
            elif w['is_adj'] or w['is_det'] or w['is_participle']:
                # Nominalize sıfat-fiil: has_poss + non-Nom case → isim gibi
                # "söylediklerini" (DIK+P3sg+Acc) → Belirtili Nesne
                # "gördüğüne" (DIK+P3sg+Dat) → Dolaylı Tümleç
                if w['is_participle'] and w['has_poss'] and w['case'] not in ('Nom', ''):
                    case = w['case']
                    if case == 'Acc':
                        w['oge'] = 'Belirtili Nesne'
                    elif case in ('Dat', 'Loc', 'Abl'):
                        w['oge'] = 'Dolaylı Tümleç'
                    elif case == 'Ins':
                        w['oge'] = 'Zarf Tümleci'
                    else:
                        w['oge'] = 'Özne'
                else:
                    w['oge'] = '?'  # çözülmemiş niteleyici
            else:
                w['oge'] = '—'

        # ── Yardımcı: '?' zincirini atlayarak ileri bak ──
        def look_ahead_role(start):
            """start indeksinden itibaren '?' ve '—' atlanarak ilk geçerli rolü döndür."""
            for j in range(start, n):
                if words[j]['oge'] not in ('?', '—'):
                    return words[j]['oge']
            return None

        # ── Pass 2: Geriye yayılma (sağdan sola) ──
        for i in range(n - 2, -1, -1):
            w = words[i]
            nxt = words[i + 1]
            nxt_role = nxt['oge']

            # Yüklem ve noktalama sabit
            if w['oge'] in ('Yüklem', 'Edat Tümleci'):
                continue
            # '—' (bağlaç/noktalama) sabit ama yayılmayı durdurmaz
            if w['oge'] == '—':
                continue

            # Gerçek hedef rolü: '?' veya '—' ise ileri bak
            if nxt_role in ('Yüklem',):
                continue
            target = nxt_role if nxt_role not in ('?', '—') else look_ahead_role(i + 1)
            if target is None or target in ('Yüklem', '—'):
                continue

            # Adj / Det / Sıfat-fiil ('?') → hedef rolü al
            if w['oge'] == '?':
                w['oge'] = target
                continue

            # Mastar + edat → rol belirleme
            # [fiilimsi + için] = amaç bildiren Zarf Tümleci (niçin?)
            # [isim + için] = Edat Tümleci
            if w['is_infinitive'] and nxt['is_postp']:
                postp_word = nxt['word'].lower()
                if postp_word == 'için':
                    w['oge'] = 'Zarf Tümleci'   # amaç bildiriyor
                else:
                    w['oge'] = 'Edat Tümleci'
                continue

            # Nom isim + Mastar → mastarın öbeğine dahil
            if w['oge'] in ('Özne', 'Belirtili Nesne') and w['pos'] in ('Noun', 'Pron', 'Num'):
                if nxt.get('is_infinitive', False) and target in ('Edat Tümleci', 'Zarf Tümleci', 'Dolaylı Tümleç'):
                    w['oge'] = target
                    continue

            # Sıfat-fiil öncesindeki her rol → baş ismin rolünü al
            # "okuldan dönmeyen" → okuldan, "hazırlıklarını tamamlayan" → hazırlıklarını
            # nxt doğrudan sıfat-fiil VEYA '?' zinciri varsa
            if nxt['is_participle'] or (nxt['oge'] == '?' and look_ahead_role(i + 1) is not None):
                if w['oge'] in ('Dolaylı Tümleç', 'Zarf Tümleci', 'Belirtili Nesne', 'Özne'):
                    w['oge'] = target
                    continue

            # Nom/Gen isim → P3sg iyelikli isim → belirtili/belirtisiz tamlama
            if w['oge'] == 'Özne' and w['pos'] in ('Noun', 'Pron', 'Num'):
                if w['case'] in ('Nom', 'Gen'):
                    if nxt['pos'] in ('Noun', 'Pron', 'Num') and nxt['has_poss']:
                        w['oge'] = target
                        continue

            # Belirtisiz tamlama: çıplak Nom (özel isim değil) + türev Acc
            if (w['oge'] == 'Özne' and w['pos'] == 'Noun' and
                    not w['has_poss'] and w['case'] == 'Nom' and
                    not w.get('is_proper', False)):
                if (nxt['pos'] == 'Noun' and nxt['case'] == 'Acc' and
                        not nxt['has_poss'] and nxt.get('has_derivation', False)):
                    w['oge'] = target

        # Kalan '?' niteleyicileri çöz (ileri bakarak)
        # Eğer look_ahead boş gelirse (cümle sonundaki sıfat) → sıfır-kopula yüklemi
        for i in range(n):
            if words[i]['oge'] == '?':
                role = look_ahead_role(i + 1)
                words[i]['oge'] = role if role else 'Yüklem'

        # ── Pass 2.5: Sıfat-fiil öbeği büтünleştirme ──
        # "Sabah erkenden kalkıp bütün hazırlıklarını tamamlayan genç öğretmen"
        # → tamamlayan'a kadar her şey öğretmen'in rolünü alır
        for p_idx in range(n):
            if not words[p_idx]['is_participle']:
                continue
            # Nominalizasyon kontrolü: has_poss + non-Nom case → isim görevi
            # "söylediklerini" (DIK+P3sg+Acc) sıfat-fiil öbeği başlatmaz
            pw = words[p_idx]
            if pw['has_poss'] and pw['case'] not in ('Nom', ''):
                continue
            # Sıfat-fiilin sağındaki baş ismi bul
            head_role = None
            for j in range(p_idx + 1, min(p_idx + 4, n)):
                if words[j]['pos'] in ('Noun', 'Pron') and words[j]['oge'] not in ('?', '—'):
                    head_role = words[j]['oge']
                    break
                if words[j]['oge'] == 'Yüklem':
                    break
            if head_role is None:
                continue
            # Sol sınır: önceki Yüklem veya cümle başı
            left = 0
            for k in range(p_idx - 1, -1, -1):
                if words[k]['oge'] == 'Yüklem':
                    left = k + 1
                    break
            # Sol sınırdan sıfat-file kadar her şeyi baş ismin rolüne çek
            for k in range(left, p_idx + 1):
                if words[k]['oge'] not in ('—', 'Yüklem', 'Edat Tümleci'):
                    words[k]['oge'] = head_role

        # ── Pass 2.7: "için" amaç tümleci geriye yayılması ──
        # Mastar + için → Zarf Tümleci; isim + için → Edat Tümleci
        for i in range(n):
            if not (words[i]['is_postp'] and words[i]['word'].lower() in ('için', 'dolayı', 'üzerine', 'kadar', 'karşı')):
                continue
            postp_word = words[i]['word'].lower()
            # için'den hemen önce mastar var mı?
            has_infinitive_before = any(
                words[k].get('is_infinitive', False)
                for k in range(max(0, i - 3), i)
            )
            grup_rol = 'Zarf Tümleci' if (postp_word == 'için' and has_infinitive_before) else 'Edat Tümleci'
            words[i]['oge'] = grup_rol   # postp'un kendisi de aynı rolü alır
            for k in range(i - 1, -1, -1):
                if words[k]['oge'] in ('Yüklem', 'Özne'):
                    break
                if words[k]['oge'] == '—':
                    break
                words[k]['oge'] = grup_rol

        # ── Pass 3: Belirtisiz Nesne tespiti ──
        verb_idx = None
        for i in range(n - 1, -1, -1):
            if words[i]['is_verb']:
                verb_idx = i
                break

        if verb_idx is not None:
            v_person = words[verb_idx].get('verb_person', 3)

            ozne_groups = []
            cur = []
            for i, w in enumerate(words):
                if w['oge'] == 'Özne':
                    cur.append(i)
                elif w['oge'] == '—':
                    # Bağlaç/noktalama Özne grubunu bölmez — atla
                    pass
                else:
                    if cur:
                        ozne_groups.append(cur)
                        cur = []
            if cur:
                ozne_groups.append(cur)

            if ozne_groups:
                last_grp = ozne_groups[-1]
                before_verb = all(
                    words[j]['oge'] in ('—',)
                    for j in range(last_grp[-1] + 1, verb_idx)
                )
                if before_verb:
                    if len(ozne_groups) > 1:
                        for idx in last_grp:
                            words[idx]['oge'] = 'Belirtisiz Nesne'
                    elif v_person in (1, 2):
                        for idx in last_grp:
                            words[idx]['oge'] = 'Belirtisiz Nesne'

        # ── Pass 4: Gizli yüklem güvencesi ──
        # Hiç Yüklem yoksa son anlamlı kelimeyi Yüklem yap
        if not any(w['oge'] == 'Yüklem' for w in words):
            for i in range(n - 1, -1, -1):
                if words[i]['oge'] not in ('—',):
                    words[i]['oge'] = 'Yüklem'
                    break

        return [{'word': w['word'], 'oge': w['oge'], 'is_postp': w['is_postp']} for w in words]

    except Exception as ex:
        logger.warning(f"cumle_ogelerini_bul hata: {ex}")
        return []

app = Flask(__name__, static_folder='static')
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# İngilizce → Türkçe Dilbilgisi Etiket Çevirisi
# ─────────────────────────────────────────────

# Sözcük türleri (POS tags)
POS_TR = {
    "Noun": "İsim",
    "Verb": "Fiil",
    "Adj": "Sıfat",
    "Adv": "Zarf",
    "Det": "Belirteç",
    "Conj": "Bağlaç",
    "Pron": "Zamir",
    "Postp": "Edat",
    "Punc": "Noktalama",
    "Ques": "Soru",
    "Interj": "Ünlem",
    "Num": "Sayı",
    "Dup": "İkileme",
}

# Alt türler / Özellikler
SECONDARY_POS_TR = {
    "Prop": "Özel",
    "Time": "Zaman",
    "Quant": "Nicelik",
    "PCNom": "Yalın",
    "PCAcc": "Belirtme",
    "PCDat": "Yönelme",
    "PCAbl": "Ayrılma",
    "PCIns": "Araçlık",
    "PCGen": "Tamlayan",
    "Pers": "Kişi",
    "Demons": "İşaret",
    "Reflex": "Dönüşlü",
}

# Kişi/sayı ekleri
AGREEMENT_TR = {
    "A1sg": "1.Tekil",
    "A2sg": "2.Tekil",
    "A3sg": "3.Tekil",
    "A1pl": "1.Çoğul",
    "A2pl": "2.Çoğul",
    "A3pl": "3.Çoğul",
}

# İyelik ekleri
POSSESSIVE_TR = {
    "P1sg": "1.Tekil İyelik",
    "P2sg": "2.Tekil İyelik",
    "P3sg": "3.Tekil İyelik",
    "P1pl": "1.Çoğul İyelik",
    "P2pl": "2.Çoğul İyelik",
    "P3pl": "3.Çoğul İyelik",
    "Pnon": "",
}

# Durum ekleri (Hal ekleri)
CASE_TR = {
    "Nom": "Yalın",
    "Acc": "Belirtme (-i)",
    "Gen": "Tamlayan (-in)",
    "Dat": "Yönelme (-e)",
    "Loc": "Bulunma (-de)",
    "Abl": "Ayrılma (-den)",
    "Ins": "Araçlık (-le)",
    "Equ": "Eşitlik (-ce)",
}

# Zaman / Kip ekleri
TENSE_MOOD_TR = {
    "Pres": "Geniş Zaman",
    "Past": "Geçmiş Zaman (-di)",
    "Narr": "Duyulan Geçmiş (-miş)",
    "Fut": "Gelecek Zaman (-ecek)",
    "Aor": "Geniş Zaman (-r)",
    "Prog1": "Şimdiki Zaman (-yor)",
    "Prog2": "Şimdiki Zaman (-mekte)",
    "Imp": "Emir",
    "Opt": "İstek (-e)",
    "Cond": "Şart (-se)",
    "Neces": "Gereklilik (-meli)",
    "Desr": "Dilek (-se)",
    "Cop": "Ek-Fiil",
    "Able": "Yeterlik (-ebil)",
    "Unable": "Yetersizlik (-eme)",
    "Pass": "Edilgen (-il)",
    "Caus": "Ettirgen (-tir)",
    "Recip": "İşteş (-iş)",
    "Reflex": "Dönüşlü (-in)",
    "Neg": "Olumsuz (-me)",
    "NegPart": "Olumsuzluk",
}

# Sıfat-fiil / Zarf-fiil ekleri
PARTICIPLE_TR = {
    "FutPart": "Sıfat-Fiil (-ecek)",
    "PastPart": "Sıfat-Fiil (-dik/-miş)",
    "PresPart": "Sıfat-Fiil (-en)",
    "AorPart": "Sıfat-Fiil (-r/-mez)",
    "NarrPart": "Sıfat-Fiil (-miş)",
    "FeelLike": "İstek (-esi)",
    "NotState": "Olumsuzluk Durumu",
    "ActOf": "Eylem İsmi",
    "WithoutDoing": "Zarf-Fiil (-meden)",
    "While": "Zarf-Fiil (-ken)",
    "When": "Zarf-Fiil (-ince)",
    "SinceDoingSo": "Zarf-Fiil (-eli)",
    "ByDoingSo": "Zarf-Fiil (-erek)",
    "AfterDoingSo": "Zarf-Fiil (-ip)",
    "AsLongAs": "Zarf-Fiil (-dikçe)",
    "Inf1": "Mastar (-mek)",
    "Inf2": "Mastar (-me)",
    "Inf3": "Mastar (-iş)",
}

# Diğer ekler
OTHER_TR = {
    "Zero": "Sıfır Ek",
    "Become": "Oluş (-leş)",
    "Ly": "Zarf (-ce)",
    "With": "Taşıma (-li)",
    "Without": "Yokluk (-siz)",
    "Ness": "İsimleştirme (-lik)",
    "Rel": "İlgi (-ki)",
    "Agt": "Yapan (-ci)",
    "Dim": "Küçültme (-cik)",
    "Related": "İlişkili (-sal)",
    "JustLike": "Benzerlik (-imsi)",
    "Quite": "Oldukça (-ce)",
    "Resemb": "Benzetme (-msı)",
}

# Yapım eki etiketleri (Derivational morpheme tags)
# Gövde token çıkarımında kullanılır
DERIVATIONAL_MORPHEMES = {
    # İsim yapım ekleri
    'Ness',      # -lık/-lik/-luk/-lük (isimleştirme)
    'Agt',       # -cı/-ci/-cu/-cü (yapan/ajan)
    'Dim',       # -cık/-cik/-cuk/-cük (küçültme)
    'ActOf',     # eylem ismi
    # Sıfat yapım ekleri
    'With',      # -lı/-li/-lu/-lü
    'Without',   # -sız/-siz/-suz/-süz
    'Related',   # -sal/-sel
    'JustLike',  # -ımsı/-imsi
    'Resemb',    # -msı
    'Rel',       # -ki
    # Fiil yapım ekleri (çatı ekleri)
    'Become',    # -laş/-leş (oluş)
    'Pass',      # -ıl/-il (edilgen)
    'Caus',      # -tır/-tir (ettirgen)
    'Recip',     # -ış/-iş (işteş)
    'Reflex',    # -ın/-in (dönüşlü)
    'Able',      # -abil/-ebil (yeterlik)
    'Unable',    # -ama/-eme (yetersizlik)
    # Zarf yapım ekleri
    'Ly',        # -ca/-ce
    'Quite',     # -ca/-ce (oldukça)
    # Sıfat-fiil ekleri
    'FutPart', 'PastPart', 'PresPart', 'AorPart', 'NarrPart',
    'FeelLike', 'NotState',
    # Zarf-fiil ekleri
    'WithoutDoing', 'While', 'When', 'SinceDoingSo',
    'ByDoingSo', 'AfterDoingSo', 'AsLongAs',
    # Mastar ekleri
    'Inf1', 'Inf2', 'Inf3',
}


def extract_stem_token(raw_analysis):
    """
    Ham Zemberek analiz stringinden gövde token'ını çıkarır.
    Sözlük/lemma formunu kullanır (yüzey biçimi değil).

    Kurallar:
    1. Tüm çekim eklerini kaldırır.
    2. Son yapım ekine kadar olan kısmı korur.
    3. Yapım eki yoksa sözlük kökünü döndürür.
    4. Ünsüz yumuşaması (t→d, p→b, k→g, ç→c) düzeltilir:
       Sözlük formu kullanılır (gid→git, kitab→kitap).

    Örnek: "[gitmek:Verb] gid:Verb+iyor:Prog1+A3sg" → "git"
    Örnek: "[güzel:Adj] güzel:Adj+leş:Become→Verb+ti:Past+A3sg" → "güzelleş"
    """
    try:
        # Köşeli parantezden sözlük kökü (lemma) ve POS'u çıkar
        bracket_match = re.match(r'\[([^\]]+)\]', raw_analysis)
        lemma_root = ''

        if bracket_match:
            bracket_content = bracket_match.group(1)
            if ':' in bracket_content:
                lemma = bracket_content.split(':')[0].strip()
                pos_part = bracket_content.split(':')[1].strip()
                pos = pos_part.split(',')[0].strip()

                # Fiiller için mastar ekini (-mek/-mak) kaldır
                if pos == 'Verb':
                    if lemma.endswith('mek') or lemma.endswith('mak'):
                        lemma_root = lemma[:-3]
                    else:
                        lemma_root = lemma
                else:
                    lemma_root = lemma
            else:
                lemma_root = bracket_content.strip()

        # Köşeli parantez sonrası suffix kısmını al
        bracket_end = raw_analysis.rfind(']')
        if bracket_end >= 0:
            suffix_part = raw_analysis[bracket_end + 1:].strip()
        else:
            suffix_part = raw_analysis.strip()

        if not suffix_part:
            return lemma_root or raw_analysis.strip()

        # '|' karakterini '+' ile değiştir (çekim grubu ayırıcısı)
        suffix_part = suffix_part.replace('|', '+')

        # '+' ile segmentlere ayır
        segments = suffix_part.split('+')

        morphemes = []
        is_first_surface = True
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            surface = ''
            tag = seg

            if ':' in seg:
                colon_idx = seg.index(':')
                surface = seg[:colon_idx]
                tag = seg[colon_idx + 1:]

            # Base tag'ı çıkar (→ ve | öncesi)
            base_tag = tag
            if '→' in base_tag:
                base_tag = base_tag.split('→')[0]
            if '|' in base_tag:
                base_tag = base_tag.split('|')[0]

            is_deriv = base_tag in DERIVATIONAL_MORPHEMES

            # İlk yüzey biçimli morfem (kök) için sözlük formunu kullan
            # Bu, ünsüz yumuşaması düzeltmesini sağlar (gid→git, kitab→kitap)
            if is_first_surface and surface and lemma_root:
                surface = lemma_root
                is_first_surface = False
            elif surface:
                is_first_surface = False

            morphemes.append({
                'surface': surface,
                'is_deriv': is_deriv,
            })

        # Son yapım ekinin indeksini bul (yüzey biçimi olan)
        last_deriv_idx = -1
        for i, m in enumerate(morphemes):
            if m['is_deriv'] and m['surface']:
                last_deriv_idx = i

        if last_deriv_idx == -1:
            # Yapım eki yok → sözlük kökünü döndür
            return lemma_root or (morphemes[0]['surface'] if morphemes else raw_analysis.strip())

        # Kökten son yapım ekine kadar yüzey biçimlerini birleştir
        stem = ''
        for i in range(last_deriv_idx + 1):
            stem += morphemes[i]['surface']

        return stem if stem else (lemma_root or raw_analysis.strip())

    except Exception:
        # Hata durumunda köşeli parantezden kökü çıkarmayı dene
        match = re.match(r'\[([^:]+):', raw_analysis)
        return match.group(1) if match else raw_analysis.strip()


# Token tip çevirileri
TOKEN_TYPE_TR = {
    "Word": "Kelime",
    "Punctuation": "Noktalama",
    "Number": "Sayı",
    "Time": "Saat",
    "Date": "Tarih",
    "Email": "E-posta",
    "URL": "URL",
    "Mention": "Bahsetme",
    "HashTag": "Etiket",
    "Emoticon": "Emoji",
    "RomanNumeral": "Romen Rakamı",
    "Abbreviation": "Kısaltma",
    "Percent": "Yüzde",
    "Unknown": "Bilinmeyen",
    "SpaceTab": "Boşluk",
    "NewLine": "Yeni Satır",
    "MetaTag": "Meta Etiket",
    "UnknownWord": "Bilinmeyen Kelime",
    "PercentNumeral": "Yüzde Sayı",
}

# Tüm etiketleri birleştir
ALL_TAGS_TR = {}
ALL_TAGS_TR.update(POS_TR)
ALL_TAGS_TR.update(SECONDARY_POS_TR)
ALL_TAGS_TR.update(AGREEMENT_TR)
ALL_TAGS_TR.update(POSSESSIVE_TR)
ALL_TAGS_TR.update(CASE_TR)
ALL_TAGS_TR.update(TENSE_MOOD_TR)
ALL_TAGS_TR.update(PARTICIPLE_TR)
ALL_TAGS_TR.update(OTHER_TR)


def translate_morph_tag(tag: str) -> str:
    """Tek bir morfolojik etiketi Türkçe'ye çevirir."""
    return ALL_TAGS_TR.get(tag, tag)


def translate_analysis_string(analysis: str) -> str:
    """
    Zemberek analiz stringini Türkçe'ye çevirir.
    Örnek giriş:  [kale:Noun, Prop] kale:Noun+A3sg+m:P1sg+in:Gen
    Örnek çıkış:  [kale:İsim, Özel] kale:İsim+3.Tekil+m:1.Tekil İyelik+in:Tamlayan (-in)
    """
    result = analysis

    # Köşeli parantez içindeki POS etiketlerini çevir
    def replace_bracket(match):
        content = match.group(1)
        parts = content.split(":")
        if len(parts) == 2:
            word = parts[0]
            tags = parts[1]
            tag_parts = [t.strip() for t in tags.split(",")]
            translated = [translate_morph_tag(t) for t in tag_parts]
            return f"[{word}:{', '.join(translated)}]"
        return match.group(0)

    result = re.sub(r'\[([^\]]+)\]', replace_bracket, result)

    # Köşeli parantez dışındaki etiketleri çevir
    # Format: kök:Etiket+ek:Etiket+ek:Etiket
    # → işlemi dışarıda yapmak lazım, parantez sonrasındaki kısmı çevir

    # Parantez dışı kısmı bul ve çevir
    bracket_end = result.rfind(']')
    if bracket_end >= 0:
        prefix = result[:bracket_end + 1]
        suffix = result[bracket_end + 1:].strip()
    else:
        prefix = ""
        suffix = result

    if suffix:
        # Her '+' ile ayrılmış parçayı çevir
        translated_parts = []
        segments = suffix.split('+')
        for seg in segments:
            # Format: "ek:Etiket" veya "kök:Etiket" veya "Etiket"
            if ':' in seg:
                sub_parts = seg.split(':')
                morph = sub_parts[0]
                tag = sub_parts[1]
                # Bileşik etiketler: "Zero→Noun" gibi
                if '→' in tag:
                    arrow_parts = tag.split('→')
                    arrow_translated = [translate_morph_tag(a) for a in arrow_parts]
                    translated_parts.append(f"{morph}:{' → '.join(arrow_translated)}")
                elif '|' in tag:
                    pipe_parts = tag.split('|')
                    pipe_translated = [translate_morph_tag(p) for p in pipe_parts]
                    translated_parts.append(f"{morph}:{'|'.join(pipe_translated)}")
                else:
                    translated_parts.append(f"{morph}:{translate_morph_tag(tag)}")
            elif '→' in seg:
                arrow_parts = seg.split('→')
                arrow_translated = [translate_morph_tag(a) for a in arrow_parts]
                translated_parts.append(' → '.join(arrow_translated))
            elif '|' in seg:
                pipe_parts = seg.split('|')
                pipe_translated = [translate_morph_tag(p) for p in pipe_parts]
                translated_parts.append('|'.join(pipe_translated))
            else:
                translated_parts.append(translate_morph_tag(seg))

        suffix_translated = ' + '.join(translated_parts)
        result = f"{prefix} {suffix_translated}" if prefix else suffix_translated

    return result


def translate_token_type(type_name: str) -> str:
    """Token tipini Türkçe'ye çevirir."""
    return TOKEN_TYPE_TR.get(type_name, type_name)


# ─────────────────────────────────────────────
# Initialize zemberek components
# ─────────────────────────────────────────────

logger.info("Zemberek NLP bileşenleri yükleniyor...")
start = time.time()

morphology = TurkishMorphology.create_with_defaults()
normalizer = TurkishSentenceNormalizer(morphology)
spell_checker = TurkishSpellChecker(morphology)
extractor = TurkishSentenceExtractor()
tokenizer = TurkishTokenizer.DEFAULT

logger.info(f"Tüm bileşenler {time.time() - start:.2f} saniyede yüklendi.")


# ─────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/normalize', methods=['POST'])
def normalize():
    """Cümle normalizasyonu"""
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': 'Metin boş olamaz'}), 400

    start = time.time()
    result = normalizer.normalize(text)
    elapsed = time.time() - start

    return jsonify({
        'input': text,
        'output': result,
        'elapsed': f"{elapsed:.4f}s"
    })


@app.route('/api/spell-check', methods=['POST'])
def spell_check():
    """Yazım kontrolü ve öneri"""
    data = request.get_json()
    word = data.get('word', '')
    if not word.strip():
        return jsonify({'error': 'Kelime boş olamaz'}), 400

    start = time.time()
    words = word.strip().split()
    results = []
    for w in words:
        suggestions = spell_checker.suggest_for_word(w)
        results.append({
            'word': w,
            'suggestions': suggestions
        })
    elapsed = time.time() - start

    return jsonify({
        'results': results,
        'elapsed': f"{elapsed:.4f}s"
    })


@app.route('/api/sentence-boundary', methods=['POST'])
def sentence_boundary():
    """Cümle sınırı tespiti"""
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': 'Metin boş olamaz'}), 400

    start = time.time()
    sentences = extractor.from_paragraph(text)
    elapsed = time.time() - start

    return jsonify({
        'sentences': list(sentences),
        'count': len(sentences),
        'elapsed': f"{elapsed:.4f}s"
    })


@app.route('/api/morphology', methods=['POST'])
def analyze_morphology():
    """Morfolojik analiz"""
    data = request.get_json()
    word = data.get('word', '')
    if not word.strip():
        return jsonify({'error': 'Kelime boş olamaz'}), 400

    start = time.time()
    results = morphology.analyze(word.strip())
    analyses_raw = [str(r) for r in results]
    analyses_tr = [translate_analysis_string(a) for a in analyses_raw]

    # Gövde token'larını çıkar
    stem_tokens = [extract_stem_token(raw) for raw in analyses_raw]
    stem_token = stem_tokens[0] if stem_tokens else word.strip()

    elapsed = time.time() - start

    return jsonify({
        'word': word.strip(),
        'analyses': analyses_tr,
        'analyses_raw': analyses_raw,
        'stem_token': stem_token,
        'stem_tokens': stem_tokens,
        'count': len(analyses_tr),
        'elapsed': f"{elapsed:.4f}s"
    })


@app.route('/api/disambiguate', methods=['POST'])
def disambiguate():
    """Cümle analizi ve anlam ayrıştırma"""
    data = request.get_json()
    sentence = data.get('sentence', '')
    if not sentence.strip():
        return jsonify({'error': 'Cümle boş olamaz'}), 400

    start = time.time()
    analysis = morphology.analyze_sentence(sentence)
    after = morphology.disambiguate(sentence, analysis)

    before_results = []
    for e in analysis:
        word_analyses = []
        for s in e:
            raw = s.format_string()
            word_analyses.append(translate_analysis_string(raw))
        before_results.append({
            'word': e.inp,
            'analyses': word_analyses
        })

    best = after.best_analysis()
    after_raw = []
    
    for i, s in enumerate(best):
        raw = s.format_string()
        word = analysis[i].inp
        
        # Özel İsim (Proper Noun) düzeltmesi
        # Zemberek bazen "Kadir" gibi kelimeleri sıfat (Adj) olarak ayırabilir.
        # Büyük harfle başlayan kelimeleri İsme (Noun) zorla.
        if word and word[0].isupper():
            if ':Noun' not in raw:
                noun_analyses = [a for a in analysis[i] if ':Noun' in a.format_string()]
                if noun_analyses:
                    prop_analyses = [a for a in noun_analyses if 'Prop' in a.format_string()]
                    if prop_analyses:
                        raw = prop_analyses[0].format_string()
                    elif ':Adj' in raw:
                        raw = noun_analyses[0].format_string()
                        
        after_raw.append(raw)
        
    after_results = [translate_analysis_string(raw) for raw in after_raw]

    # Gövde token'larını çıkar
    words = [e.inp for e in analysis]
    stem_tokens = [extract_stem_token(raw) for raw in after_raw]
    word_stems = []
    for i, w in enumerate(words):
        st = stem_tokens[i] if i < len(stem_tokens) else w
        word_stems.append({'word': w, 'stem': st})

    elapsed = time.time() - start

    return jsonify({
        'sentence': sentence,
        'before': before_results,
        'after': after_results,
        'word_stems': word_stems,
        'elapsed': f"{elapsed:.4f}s"
    })


@app.route('/api/tokenize', methods=['POST'])
def tokenize_text():
    """Tokenizasyon"""
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': 'Metin boş olamaz'}), 400

    start = time.time()
    tokens = tokenizer.tokenize(text)
    token_list = []
    for token in tokens:
        type_name = token.type_.name
        token_list.append({
            'content': token.content,
            'type': translate_token_type(type_name),
            'type_raw': type_name,
            'start': token.start,
            'end': token.end
        })
    elapsed = time.time() - start

    return jsonify({
        'tokens': token_list,
        'count': len(token_list),
        'elapsed': f"{elapsed:.4f}s"
    })


@app.route('/api/cumle-ogeleri', methods=['POST'])
def cumle_ogeleri():
    """Cümlenin ögeleri analizi — kelime gruplarıyla"""
    data = request.get_json()
    sentence = data.get('sentence', '')
    if not sentence.strip():
        return jsonify({'error': 'Cümle boş olamaz'}), 400

    start = time.time()

    ogeler = cumle_ogelerini_bul(sentence.strip(), morphology)

    # Ardışık aynı ögeye sahip kelimeleri grupla
    # '—' (bağlaç "ama", "ve" vb.) aynı roldeki kelimeleri birleştirirken dahil edilir
    groups = []
    if ogeler:
        cur_oge = None
        cur_words = []
        pending_dashes = []  # bekleyen '—' kelimeler

        for item in ogeler:
            if item['oge'] == '—':
                # Beklemeye al — sonraki kelimenin rolüne göre karar verilecek
                pending_dashes.append(item['word'])
            else:
                if cur_oge is None:
                    # İlk anlamlı kelime
                    cur_oge = item['oge']
                    cur_words = [item['word']]
                    pending_dashes = []
                elif item['oge'] == cur_oge:
                    # Aynı rol: bekleyen '—'leri de gruba ekle
                    cur_words.extend(pending_dashes)
                    cur_words.append(item['word'])
                    pending_dashes = []
                    # Postp (için, gibi...) grubu kapatır — sonraki kelime yeni grup
                    if item.get('is_postp', False):
                        groups.append({'phrase': ' '.join(cur_words), 'oge': cur_oge})
                        cur_oge = None
                        cur_words = []
                else:
                    # Farklı rol: mevcut grubu kaydet, yeni başlat
                    groups.append({'phrase': ' '.join(cur_words), 'oge': cur_oge})
                    cur_oge = item['oge']
                    cur_words = [item['word']]
                    pending_dashes = []

        if cur_words and cur_oge:
            groups.append({'phrase': ' '.join(cur_words), 'oge': cur_oge})

    elapsed = time.time() - start

    return jsonify({
        'sentence': sentence,
        'groups': groups,
        'count': len([g for g in groups if g['oge'] != '—']),
        'elapsed': f"{elapsed:.4f}s"
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  🇹🇷 Zemberek NLP Web Uygulaması")
    print("  http://localhost:8080 adresinde çalışıyor")
    print("=" * 60 + "\n")
    app.run(debug=False, port=8080, host='0.0.0.0')
