from util.data.text_encoder import SubwordTextEncoder
from util.data import text_encoder

from collections import Counter

dict = {}
for line in open('/Users/zhaosanqiang916/git/keyphrase_data/kp20k2/abstr.voc'):
    items = line.split('\t')
    word = items[0]
    cnt = int(items[1])
    dict[word] = cnt

c = Counter(dict)
output_path = "/Users/zhaosanqiang916/git/keyphrase_data/kp20k2/abstr.subvoc"
sub_word = SubwordTextEncoder.build_to_target_size(30000, c, 1, 1e3,
                                                               num_iterations=100)
for i, subtoken_string in enumerate(sub_word._all_subtoken_strings):
    if subtoken_string in text_encoder.RESERVED_TOKENS_DICT:
        sub_word._all_subtoken_strings[i] = subtoken_string + "_"
sub_word.store_to_file(output_path)