import json
import spacy

nlp = spacy.load('en')

def generate(input_file, output_file):
    f = open(output_file, 'w')
    lines = []
    for line in open(input_file):
        obj = json.loads(line.lower())
        title = ' '.join([w.text for w in list(nlp(obj['title']))])
        abstract = ' '.join([w.text for w in list(nlp(obj['abstract']))])
        kphrases = []
        for kphrase in obj['keyword'].split(';'):
            kphrase = ' '.join([w.text for w in list(nlp(kphrase))])
            kphrases.append(kphrase)
        kphrases = ';'.join(kphrases)

        obj = {}
        obj['title'] = title
        obj['abstract'] = abstract
        obj['kphrases'] = kphrases
        line = json.dumps(obj)
        lines.append(line)
        # if len(lines) >= 1000:
        #     f.write('\n'.join(lines))
        #     lines.clear()
    f.write('\n'.join(lines))
    lines.clear()
    f.close()

input_file_list = ['/Users/zhaosanqiang916/git/keyphrase_data/kp20k/ke20k_training.json',
             '/Users/zhaosanqiang916/git/keyphrase_data/kp20k/ke20k_validation.json',
             '/Users/zhaosanqiang916/git/keyphrase_data/kp20k/ke20k_testing.json']
output_file_list = ['/Users/zhaosanqiang916/git/keyphrase_data/kp20k/ke20k_training.processed.json',
             '/Users/zhaosanqiang916/git/keyphrase_data/kp20k/ke20k_validation.processed.json',
             '/Users/zhaosanqiang916/git/keyphrase_data/kp20k/ke20k_testing.processed.json']
for i in range(3):
    generate(input_file_list[i], output_file_list[i])