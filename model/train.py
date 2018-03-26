# For fix slurm cannot load PYTHONPATH
import sys

sys.path.insert(0, '/ihome/cs2770_s2018/maz54/kp/keyphrase')


import numpy as np
from data_generator.train_data import TrainData
from model.graph import Graph
import tensorflow as tf
from datetime import datetime
import random as rd
from model.model_config import list_config
import tensorflow.contrib.slim as slim
from os.path import exists, dirname
from os import listdir


def get_feed(objs, data, model_config):
    input_feed = {}
    for obj in objs:
        tmp_abstr, tmp_kword = [], []
        for i in range(model_config.batch_size):
            data_sample = data.get_data_sample()
            assert len(data_sample['abstr']) == model_config.max_abstr_len
            tmp_abstr.append(data_sample['abstr'])
            if len(data_sample['kwords']) >= model_config.max_cnt_kword:
                tmp_kword.append(rd.sample(data_sample['kwords'], model_config.max_cnt_kword))
            else:
                kwords_tmp = data_sample['kwords']
                # for _ in range(int(model_config.max_cnt_kword / len(data_sample['kwords']))):
                #     kwords_tmp += data_sample['kwords']
                while len(kwords_tmp) < model_config.max_cnt_kword:
                    kwords_tmp.append([1] * len(kwords_tmp[0]))
                tmp_kword.append(kwords_tmp)

        for step in range(model_config.max_abstr_len):
            input_feed[obj['abstr_ph'][step].name] = [
                tmp_abstr[batch_idx][step] for batch_idx in range(model_config.batch_size)]

        for kword_idx in range(model_config.max_cnt_kword):
            for step in range(model_config.max_kword_len):
                input_feed[obj['kwords_ph'][kword_idx][step].name] = [
                    tmp_kword[batch_idx][kword_idx][step] for batch_idx in range(model_config.batch_size)]
    return input_feed


def find_best_ckpt(model_config):
    if not exists(model_config.warm_start):
        dir = dirname(model_config.warm_start)
        files = listdir(dir)
        max_id = -1
        for file in files:
            if file.startswith('model.ckpt-') and file.endswith('.meta'):
                sid = file.index('model.ckpt-') + len('model.ckpt-')
                eid = file.rindex('.')
                id = int(file[sid:eid])
                max_id = max(id, max_id)
        return ''.join([dir, '/model.ckpt-', str(max_id)])
    else:
        return model_config.warm_start



def train(model_config):
    traindata = TrainData(model_config)
    graph = Graph(model_config, True)
    graph.create_model_multigpu()

    if model_config.warm_start:
        model_config.warm_start = find_best_ckpt(model_config)
        ckpt_path = model_config.warm_start
        var_list = slim.get_variables_to_restore()
        available_vars = {}
        reader = tf.train.NewCheckpointReader(ckpt_path)
        var_dict = {var.op.name: var for var in var_list}
        for var in var_dict:
            if 'global_step' in var:
                continue
            if 'optimization' in var:
                continue
            if reader.has_tensor(var):
                var_ckpt = reader.get_tensor(var)
                var_cur = var_dict[var]
                if any([var_cur.shape[i] != var_ckpt.shape[i] for i in range(len(var_ckpt.shape))]):
                    print('Variable %s missing due to shape.', var)
                else:
                    available_vars[var] = var_dict[var]
            else:
                print('Variable %s missing.', var)

        partial_restore_ckpt = slim.assign_from_checkpoint_fn(
            ckpt_path, available_vars,
            ignore_missing_vars=False, reshape_variables=False)

    def init_fn(session):
        if model_config.warm_start:
            partial_restore_ckpt(session)
            print('Restore Ckpt %s.' % model_config.warm_start)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sv = tf.train.Supervisor(logdir=model_config.logdir,
                             global_step=graph.global_step,
                             saver=graph.saver,
                             save_model_secs=600,
                             init_fn=init_fn)
    sess = sv.PrepareSession(config=config)
    perplexitys = []
    start_time = datetime.now()
    while True:
        input_feed = get_feed(graph.objs, traindata, model_config)
        # fetches = [graph.train_op, graph.loss, graph.global_step, graph.perplexity, graph.objs[0]['targets'], graph.objs[0]['attn_stick']]
        # _, loss, step, perplexity, target, attn_stick = sess.run(fetches, input_feed)
        fetches = [graph.train_op, graph.loss, graph.global_step, graph.perplexity]
        _, loss, step, perplexity = sess.run(fetches, input_feed)

        perplexitys.append(perplexity)

        if step % model_config.model_print_freq == 0:
            end_time = datetime.now()
            time_span = end_time - start_time
            start_time = end_time
            print('Perplexity:\t%f at step %d using %s.' % (np.mean(perplexitys), step, time_span))
            perplexitys.clear()
            # if model_config.subword_vocab_size > 0:
            #     print(traindata.voc_kword.describe(target[0][0]))
            # else:
            #     print(' '.join([traindata.voc_kword.describe(w) for w in target[0][0]]))


if __name__ == '__main__':
    from model.model_config import DefaultConfig, DummyConfig
    model_config = DefaultConfig()
    print(list_config(model_config))
    train(model_config)
