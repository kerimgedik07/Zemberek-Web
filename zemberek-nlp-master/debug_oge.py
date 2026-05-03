"""Cümle ögesi debug — Zemberek'in gerçek çıktısını gösterir."""
from zemberek import TurkishMorphology
import re

morphology = TurkishMorphology.create_with_defaults()

sentence = "Ege bölgesi muhteşem bir doğa olayına ev sahipliği yapıyor."
analysis = morphology.analyze_sentence(sentence)
after = morphology.disambiguate(sentence, analysis)
best = after.best_analysis()

print(f"Cümle: {sentence}\n")
print(f"{'Kelime':<15} {'POS':<8} {'Case':<6} {'Poss':<6} {'Raw Analiz'}")
print("-" * 100)

for i, s in enumerate(best):
    raw = s.format_string()
    word = analysis[i].inp

    pos_m = re.search(r':(Noun|Verb|Adj|Adv|Pron|Num|Conj|Postp|Punc|Ques|Interj|Dup|Det)', raw)
    pos = pos_m.group(1) if pos_m else '?'

    case_m = re.search(r':(Nom|Acc|Gen|Dat|Loc|Abl|Ins|Equ)', raw)
    case = case_m.group(1) if case_m else '-'

    poss_m = re.search(r'P([123])(sg|pl)', raw)
    poss = poss_m.group(0) if poss_m else 'Pnon'

    print(f"{word:<15} {pos:<8} {case:<6} {poss:<6} {raw}")
