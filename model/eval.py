# For fix slurm cannot load PYTHONPATH
import sys

sys.path.insert(0, '/ihome/cs2770_s2018/maz54/kp/keyphrase')


import numpy as np
from data_generator.eval_data import EvalData
from model.graph import Graph
import tensorflow as tf
from datetime import datetime
import random as rd
from util.checkpoint import copy_ckpt_to_modeldir
from util.decode import decode_keyphrases, decode_gt_keyphrase
from util.f1 import calculate_f1
import time
from os.path import exists
from os import mkdir
import tensorflow.contrib.slim as slim
from os import remove


def get_feed(objs, it, model_config):
    input_feed = {}
    assert len(objs) == 1
    obj = objs[0]
    tmp_abstr, tmp_kword = [], []
    is_finished = False
    for i in range(model_config.batch_size):
        data_sample = next(it)
        if data_sample is None:
            is_finished = True
            continue
        assert len(data_sample['abstr']) == model_config.max_abstr_len
        tmp_abstr.append(data_sample['abstr'])
        tmp_kword.append(data_sample['kwords'])

        for step in range(model_config.max_abstr_len):
            input_feed[obj['abstr_ph'][step].name] = [
                tmp_abstr[batch_idx][step] for batch_idx in range(len(tmp_abstr))]

    return input_feed, tmp_kword, is_finished


def eval(model_config, ckpt):
    evaldata = EvalData(model_config)
    it = evaldata.get_data_sample_it()
    tf.reset_default_graph()
    graph = Graph(model_config, False)
    graph.create_model_multigpu()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    restore_op = slim.assign_from_checkpoint_fn(
        ckpt, slim.get_variables_to_restore(),
        ignore_missing_vars=False, reshape_variables=False)
    def init_fn(session):
        restore_op\
            (session)
        # graph.saver.restore(session, ckpt)
        print('Restore ckpt:%s.' % ckpt)

    sv = tf.train.Supervisor(init_fn=init_fn)
    sess = sv.PrepareSession(config=config)
    perplexitys = []
    prec_top10s, recall_top10s, f1_top10s = [], [], []
    prec_top5s, recall_top5s, f1_top5s = [], [], []
    reports = []
    while True:
        input_feed, gt_kphrases, is_finished = get_feed(graph.objs, it, model_config)
        if is_finished:
            break
        # s_time = datetime.now()
        fetches = [graph.loss, graph.global_step, graph.perplexity, graph.objs[0]['targets'], graph.objs[0]['attn_stick']]
        loss, step, perplexity, targets, attn_stick = sess.run(fetches, input_feed)
        perplexitys.append(perplexity)

        for batch_i in range(model_config.batch_size):
            target = targets[batch_i]
            kphrases_top10 = decode_keyphrases(target, evaldata, model_config)
            kphrases_gt = decode_gt_keyphrase(gt_kphrases[batch_i], evaldata, model_config)
            kphrases_top5 = kphrases_top10[:5] if len(kphrases_top10) > 5 else kphrases_top10
            prec_top10, recall_top10, f1_top10 = calculate_f1(set(kphrases_top10), set(kphrases_gt))
            prec_top5, recall_top5, f1_top5 = calculate_f1(set(kphrases_top5), set(kphrases_gt))

            prec_top10s.append(prec_top10)
            recall_top10s.append(recall_top10)
            f1_top10s.append(f1_top10)

            prec_top5s.append(prec_top5)
            recall_top5s.append(recall_top5)
            f1_top5s.append(f1_top5)

            report = ''.join(['pred:\t', ';'.join(kphrases_top10), '\n', 'gt:\t', ';'.join(kphrases_gt)])
            reports.append(report)

        # e_time = datetime.now()
        # span = e_time - s_time
        # print('%s' % (str(span)))
    format = '%.4f'
    file_name = ''.join(['step_', str(step),
                         'f1top10_', str(format % np.mean(f1_top10s)), 'f1top5_', str(format % np.mean(f1_top5s)),
                         'prectop10_', str(format % np.mean(prec_top10s)), 'prectop5_', str(format % np.mean(prec_top5s)),
                         'recalltop10_', str(format % np.mean(recall_top10s)), 'recalltop5_', str(format % np.mean(recall_top5s)),
                         'perplexity_', str(np.mean(perplexitys))
                         ])
    if not exists(model_config.resultdir):
        mkdir(model_config.resultdir)
    f = open(model_config.resultdir + file_name, 'w')
    f.write('\n\n'.join(reports))
    f.close()

    return np.mean(f1_top10s)


def get_ckpt(modeldir, logdir, wait_second=60):
    while True:
        try:
            ckpt = copy_ckpt_to_modeldir(modeldir, logdir)
            return ckpt
        except FileNotFoundError as exp:
            if wait_second:
                print(str(exp) + '\nWait for 1 minutes.')
                time.sleep(wait_second)
            else:
                return None

if __name__ == '__main__':
    from model.model_config import DefaultConfig, DefaultValConfig, DefaultTestConfig, DummyConfig
    model_config = DefaultConfig()
    while True:
        ckpt = get_ckpt(model_config.modeldir, model_config.logdir)
        if ckpt:
            if model_config.eval_mode == 'none':
                f1 = eval(DefaultValConfig(), ckpt)
            elif model_config.eval_mode == 'truncate2000':
                f1 = eval(DefaultValConfig(), ckpt)
                # eval(DefaultTestConfig(), ckpt)
            if float(f1) < 0.39:
                remove(ckpt + '.index')
                remove(ckpt + '.meta')
                remove(ckpt + '.data-00000-of-00001')
                print('remove' + str(ckpt))
