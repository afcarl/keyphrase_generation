import json
import spacy
from datetime import datetime
from nltk.tokenize.stanford import StanfordTokenizer

tokenizer = StanfordTokenizer(path_to_jar='/Users/zhaosanqiang916/git/stanford-ner-2017-06-09/stanford-english-corenlp-2017-06-09-models.jar')
# nlp = spacy.load('en')

def generate(input_file, output_file):
    f = open(output_file, 'w')
    lines = []
    s_t = datetime.now()
    for line in open(input_file):
        obj = json.loads(line.lower())
        # title = ' '.join([w.text for w in list(nlp(obj['title']))])
        # abstract = ' '.join([w.text for w in list(nlp(obj['abstract']))])
        title = ' '.join([w for w in tokenizer.tokenize(obj['title'])])
        abstract = ' '.join([w for w in tokenizer.tokenize(obj['abstract'])])
        kphrases = []
        for kphrase in obj['keyword'].split(';'):
            # kphrase = ' '.join([w.text for w in list(nlp(kphrase))])
            kphrase = ' '.join([w for w in tokenizer.tokenize(kphrase)])
            kphrases.append(kphrase)
        kphrases = ';'.join(kphrases)

        obj = {}
        obj['title'] = title
        obj['abstract'] = abstract
        obj['kphrases'] = kphrases
        line = json.dumps(obj)
        lines.append(line)
        if len(lines) % 10 == 0:
            e_t = datetime.now()
            span_t = e_t - s_t
            print('%s use %s' % (len(lines), span_t))
            s_t = e_t
        # if len(lines) >= 1000:
        #     f.write('\n'.join(lines))
        #     lines.clear()
    f.write('\n'.join(lines))
    lines.clear()
    f.close()

input_file_list = ['/Users/zhaosanqiang916/git/keyphrase_data/kp20k/ke20k_training.json',
             '/Users/zhaosanqiang916/git/keyphrase_data/kp20k/ke20k_validation.json',
             '/Users/zhaosanqiang916/git/keyphrase_data/kp20k/ke20k_testing.json']
output_file_list = ['/Users/zhaosanqiang916/git/keyphrase_data/kp20k/ke20k_training.processed2.json',
             '/Users/zhaosanqiang916/git/keyphrase_data/kp20k/ke20k_validation.processed2.json',
             '/Users/zhaosanqiang916/git/keyphrase_data/kp20k/ke20k_testing.processed2.json']
for i in range(3):
    generate(input_file_list[i], output_file_list[i])