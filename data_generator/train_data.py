from data_generator.vocab import Vocab
from util import constant
import json
import random as rd


class TrainData:
    def __init__(self, model_config):
        self.model_config = model_config
        self.voc_abstr = Vocab(self.model_config, vocab_path=self.model_config.path_abstr_voc)
        self.voc_kword = Vocab(self.model_config, vocab_path=self.model_config.path_kword_voc)
        self.populate_data()
        self.size = len(self.data)

    def populate_data(self):
        pad_id = self.voc_abstr.encode(constant.SYMBOL_PAD)
        self.data = []
        for line in open(self.model_config.path_train_json):
            try:
                obj = json.loads(line.strip())
                if self.model_config.subword_vocab_size > 0:
                    abstr = self.voc_abstr.encode(' '.join(
                            [constant.SYMBOL_START] + obj['title'].split() + obj['abstract'].split() + [constant.SYMBOL_END]))
                    if len(abstr) > self.model_config.max_abstr_len:
                        abstr = abstr[:self.model_config.max_abstr_len]
                    else:
                        num_pad = self.model_config.max_abstr_len - len(abstr)
                        abstr.extend(num_pad * pad_id)
                    kwords = [self.voc_kword.encode(' '.join(
                        [constant.SYMBOL_START] + kphrase.split() + [constant.SYMBOL_END]))
                        for kphrase in obj['kphrases'].split(';')]
                    for kword_id, kword in enumerate(kwords):
                        if len(kword) >= self.model_config.max_kword_len:
                            kwords[kword_id] = kword[:self.model_config.max_kword_len]
                        else:
                            num_pad = self.model_config.max_kword_len - len(kword)
                            kwords[kword_id].extend(num_pad * pad_id)
                else:
                    abstr = [self.voc_abstr.encode(w) for w
                             in [constant.SYMBOL_START] + obj['title'].split() + obj['abstract'].split() + [constant.SYMBOL_END]]
                    if len(abstr) > self.model_config.max_abstr_len:
                        abstr = abstr[:self.model_config.max_abstr_len]
                    else:
                        num_pad = self.model_config.max_abstr_len - len(abstr)
                        abstr.extend(num_pad * [pad_id])

                    kwords = [[self.voc_kword.encode(w) for w
                               in [constant.SYMBOL_START] + kphrase.split() + [constant.SYMBOL_END]]
                              for kphrase in obj['kphrases'].split(';') ]
                    for kword_id, kword in enumerate(kwords):
                        if len(kword) >= self.model_config.max_kword_len:
                            kwords[kword_id] = kword[:self.model_config.max_kword_len]
                        else:
                            num_pad = self.model_config.max_kword_len - len(kword)
                            kwords[kword_id].extend(num_pad * [pad_id])
            except:
                print('json error:')


            self.data.append({
                'abstr':abstr,
                'kwords':kwords
            })

    def get_data_sample(self):
        i = rd.sample(range(self.size), 1)[0]
        return self.data[i]


