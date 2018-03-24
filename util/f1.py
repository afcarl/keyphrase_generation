def calculate_f1(kphrases_pred, kphrases_gt):
    prec = calculate_prec(kphrases_pred, kphrases_gt)
    recall = calculate_recall(kphrases_pred, kphrases_gt)
    if prec+recall == 0:
        f1 = 0
    else:
        f1 = 2 * prec * recall / (prec + recall)
    return prec, recall, f1

def calculate_prec(kphrases_pred, kphrases_gt):
    return len(kphrases_pred & kphrases_gt) / len(kphrases_pred)


def calculate_recall(kphrases_pred, kphrases_gt):
    return len(kphrases_pred & kphrases_gt) / len(kphrases_gt)