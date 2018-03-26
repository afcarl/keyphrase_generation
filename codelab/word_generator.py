import json
from collections import Counter
import spacy

# nlp = spacy.load('en')

f_abstr_voc = open('/Users/zhaosanqiang916/git/keyphrase_data/kp20k2/abstr.voc', 'w')
f_kword_voc = open('/Users/zhaosanqiang916/git/keyphrase_data/kp20k2/kword.voc', 'w')

c_abstr = Counter()
c_kword = Counter()

for line in open('/Users/zhaosanqiang916/git/keyphrase_data/kp20k2/ke20k_processed_training.json'):
    obj = json.loads(line.lower())
    # words = [w.text for w in list(nlp(obj['title']))] + [w.text for w in list(nlp(obj['abstract']))]
    words = [w for w in obj['title'].split()] + [w for w in obj['abstract'].split()]
    c_abstr.update(words)
    for kphrase in obj['kphrases'].split(';'):
        # c_kword.update([w.text for w in list(nlp(kphrase))])
        c_kword.update([w for w in kphrase.split()])


for word, cnt in c_abstr.most_common():
    f_abstr_voc.write(word)
    f_abstr_voc.write('\t')
    f_abstr_voc.write(str(cnt))
    f_abstr_voc.write('\n')
f_abstr_voc.close()


for word, cnt in c_kword.most_common():
    f_kword_voc.write(word)
    f_kword_voc.write('\t')
    f_kword_voc.write(str(cnt))
    f_kword_voc.write('\n')
f_kword_voc.close()