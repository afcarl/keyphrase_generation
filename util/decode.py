from util import constant


def decode_keyphrases(target, evaldata, model_config, topk=10):
    """Get a list of predicted keyphrases"""
    keyphrases = []
    bos_id = evaldata.voc_kword.encode('#bos#')[0]
    eos_id = evaldata.voc_kword.encode('#eos#')[0]
    # assert topk < len(target)
    for i in range(len(target)):
        if model_config.subword_vocab_size > 0:
            keyphrase = decode_keyphrase(target[i], evaldata, model_config, bos_id, eos_id)
            if keyphrase not in keyphrases:
                keyphrases.append(keyphrase)
            if len(keyphrases) >= topk:
                break
    return keyphrases


def decode_gt_keyphrase(gt_kphrases, evaldata, model_config):
    keyphrases = []
    bos_id = evaldata.voc_kword.encode('#bos#')[0]
    eos_id = evaldata.voc_kword.encode('#eos#')[0]
    for i in range(len(gt_kphrases)):
        keyphrase = decode_keyphrase(gt_kphrases[i], evaldata, model_config, bos_id, eos_id)
        keyphrases.append(keyphrase)
    return keyphrases


def decode_keyphrase(keyphrase, evaldata, model_config, bos_id=3, eos_id=4):
    """Get clean key phrase"""
    if model_config.subword_vocab_size > 0:
        keyphrase = list(keyphrase)
        if bos_id in keyphrase:
            left_idx = keyphrase.index(bos_id)+1
        else:
            left_idx = 0
        if eos_id in keyphrase:
            right_idx = keyphrase.index(eos_id)
        else:
            right_idx = len(keyphrase) - 1
        keyphrase = keyphrase[left_idx:right_idx]
        return evaldata.voc_kword.describe(keyphrase)