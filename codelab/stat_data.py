import json

path = '/Users/zhaosanqiang916/git/keyphrase_data/kp20k/ke20k_training.processed.json'
for line in open(path):
    obj = json.loads(line)
    kphrases = obj['kphrases']
    kphraseses = [kp.split() for kp in kphrases.split(';')]

    for kp in kphraseses:
        if len(kp) == 1:
            print(kphrases)

    # checker = set()
    # for kp in kphraseses:
    #     if kp[0] in checker:
    #         print(kphrases)
    #     checker.add(kp[0])