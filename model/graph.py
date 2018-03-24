import tensorflow as tf
from tensor2tensor.layers import common_attention, common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import beam_search

from util import constant
from model.loss import sequence_loss
from data_generator.vocab import Vocab

class Graph:
    def __init__(self, model_config, is_train):
        self.model_config = model_config
        self.is_train = is_train
        self.voc_abstr = Vocab(self.model_config, vocab_path=self.model_config.path_abstr_voc)
        self.voc_kword = Vocab(self.model_config, vocab_path=self.model_config.path_kword_voc)
        self.hparams = transformer.transformer_base()
        self.setup_hparams()

    def get_embedding(self):
        emb_init = tf.contrib.layers.xavier_initializer() # tf.random_uniform_initializer(-0.08, 0.08)
        emb_abstr = tf.get_variable(
            'embedding_abstr', [self.voc_abstr.vocab_size(), self.model_config.dimension], tf.float32,
            initializer=emb_init)
        emb_kword = tf.get_variable(
            'embedding_kword', [self.voc_kword.vocab_size(), self.model_config.dimension], tf.float32,
            initializer=emb_init)
        proj_w = tf.get_variable(
            'proj_w', [self.voc_kword.vocab_size(), self.model_config.dimension], tf.float32,
            initializer=emb_init)
        proj_b = tf.get_variable(
            'proj_b', shape=[self.voc_kword.vocab_size()], initializer=emb_init)
        return emb_abstr, emb_kword, proj_w, proj_b

    def embedding_fn(self, inputs, embedding):
        if type(inputs) == list:
            if not inputs:
                return []
            else:
                return [tf.nn.embedding_lookup(embedding, inp) for inp in inputs]
        else:
            return tf.nn.embedding_lookup(embedding, inputs)

    def decode_step(self, kword_input, abstr_outputs, abstr_bias, attn_stick):
        batch_go = [tf.zeros([self.model_config.batch_size, self.model_config.dimension])]
        kword_length = len(kword_input) + 1
        kword_input = tf.stack(batch_go + kword_input, axis=1)
        kword_output, new_attn_stick = self.decode_inputs_to_outputs(kword_input, abstr_outputs, abstr_bias, attn_stick)
        kword_output_list = [
            tf.squeeze(d, 1)
            for d in tf.split(kword_output, kword_length, axis=1)]
        return kword_output_list, new_attn_stick

    def decode_inputs_to_outputs(self, kword_input, abstr_outputs, abstr_bias, attn_stick):
        if self.hparams.pos == 'timing':
            kword_input = common_attention.add_timing_signal_1d(kword_input)
        kword_tribias = common_attention.attention_bias_lower_triangle(tf.shape(kword_input)[1])
        kword_input = tf.nn.dropout(
            kword_input, 1.0 - self.hparams.layer_prepostprocess_dropout)
        kword_output, new_attn_stick = transformer.transformer_decoder(
            kword_input, abstr_outputs, kword_tribias,
            abstr_bias, self.hparams, attn_stick=attn_stick)
        return kword_output, new_attn_stick

    def output_to_logit(self, prev_out, w, b):
        prev_logit = tf.add(tf.matmul(prev_out, tf.transpose(w)), b)
        return prev_logit

    def transformer_beam_search(self, abstr_outputs, abstr_bias, emb_kword, proj_w, proj_b, attn_stick=None):
        # Use Beam Search in evaluation stage
        # Update [a, b, c] to [a, a, a, b, b, b, c, c, c] if beam_search_size == 3
        encoder_beam_outputs = tf.concat(
            [tf.tile(tf.expand_dims(abstr_outputs[o, :, :], axis=0),
                     [self.model_config.beam_search_size, 1, 1])
             for o in range(self.model_config.batch_size)], axis=0)

        encoder_attn_beam_bias = tf.concat(
            [tf.tile(tf.expand_dims(abstr_bias[o, :, :, :], axis=0),
                     [self.model_config.beam_search_size, 1, 1, 1])
             for o in range(self.model_config.batch_size)], axis=0)

        def symbol_to_logits_fn(ids, attn_stick=attn_stick):
            embs = tf.nn.embedding_lookup(emb_kword, ids[:, 1:])
            embs = tf.pad(embs, [[0, 0], [1, 0], [0, 0]])
            final_outputs, new_attn_stick = self.decode_inputs_to_outputs(embs, encoder_beam_outputs, encoder_attn_beam_bias, attn_stick=attn_stick)
            return self.output_to_logit(final_outputs[:, -1, :], proj_w, proj_b), new_attn_stick

        beam_ids, beam_score, new_attn_stick = beam_search.beam_search(symbol_to_logits_fn,
                                                       tf.zeros([self.model_config.batch_size], tf.int32),
                                                       self.model_config.beam_search_size,
                                                       self.model_config.max_kword_len,
                                                       self.voc_kword.vocab_size(),
                                                       0.6,
                                                       attn_stick=attn_stick
                                                       )
        top_beam_ids = beam_ids[:, 0, 1:]
        top_beam_ids = tf.pad(top_beam_ids,
                              [[0, 0],
                               [0, self.model_config.max_kword_len - tf.shape(top_beam_ids)[1]]])
        decoder_target_list = [tf.squeeze(d, 1)
                               for d in tf.split(top_beam_ids, self.model_config.max_kword_len, axis=1)]
        decoder_score = -beam_score[:, 0] / tf.to_float(tf.shape(top_beam_ids)[1])

        return decoder_score, top_beam_ids, new_attn_stick #tf.stack(decoder_target_list, axis=1)

    def greed_search(self, id, abstr_outputs, abstr_bias, emb_kword, proj_w, proj_b, attn_stick=None):
        kword_target_tensor = tf.TensorArray(tf.int64, size=self.model_config.max_kword_len,
                                             clear_after_read=False,
                                             element_shape=[self.model_config.batch_size, ],
                                             name='kword_target_tensor_%s' % str(id))
        kword_logit_tensor = tf.TensorArray(tf.float32, size=self.model_config.max_kword_len,
                                            clear_after_read=False,
                                            element_shape=[self.model_config.batch_size,
                                                           self.voc_kword.vocab_size()],
                                            name='kword_logit_tensor_%s' % str(id))
        kword_embed_inputs_tensor = tf.TensorArray(tf.float32,
                                                   size=1, dynamic_size=True,
                                                   clear_after_read=False,
                                                   element_shape=[self.model_config.batch_size,
                                                                  self.model_config.dimension],
                                                   name='kword_embed_inputs_tensor_%s' % str(id))
        kword_output_tensor = tf.TensorArray(tf.float32, size=self.model_config.max_kword_len,
                                             clear_after_read=False,
                                             element_shape=[self.model_config.batch_size,
                                                            self.model_config.dimension],
                                             name='kword_output_tensor_%s' % str(id))

        kword_embed_inputs_tensor = kword_embed_inputs_tensor.write(
            0, tf.zeros([self.model_config.batch_size, self.model_config.dimension]))

        def _is_finished(step, kword_target_tensor, kword_logit_tensor,
                         kword_embed_inputs_tensor, kword_output_tensor, attn_stick):
            return tf.less(step, self.model_config.max_kword_len)

        def _recursive(step, kword_target_tensor, kword_logit_tensor,
                       kword_embed_inputs_tensor, kword_output_tensor, attn_stick):
            cur_kword_embed_inputs_tensor = kword_embed_inputs_tensor.stack()
            cur_kword_embed_inputs_tensor = tf.transpose(cur_kword_embed_inputs_tensor, perm=[1, 0, 2])

            kword_outputs = self.decode_inputs_to_outputs(cur_kword_embed_inputs_tensor, abstr_outputs, abstr_bias,
                                                          attn_stick=attn_stick)
            kword_output = kword_outputs[:, -1, :]

            kword_logit = self.output_to_logit(kword_output, proj_w, proj_b)
            kword_target = tf.argmax(kword_logit, output_type=tf.int64, axis=-1)
            kword_output_tensor = kword_output_tensor.write(step, kword_output)
            kword_logit_tensor = kword_logit_tensor.write(step, kword_logit)
            kword_target_tensor = kword_target_tensor.write(step, kword_target)
            kword_embed_inputs_tensor = kword_embed_inputs_tensor.write(
                step + 1, tf.nn.embedding_lookup(emb_kword, kword_target))
            return step + 1, kword_target_tensor, kword_logit_tensor, kword_embed_inputs_tensor, kword_output_tensor, attn_stick

        step = tf.constant(0)
        (_, kword_target_tensor, kword_logit_tensor, kword_embed_inputs_tensor,
         kword_output_tensor, attn_stick) = tf.while_loop(
            _is_finished, _recursive,
            [step, kword_target_tensor, kword_logit_tensor, kword_embed_inputs_tensor,
             kword_output_tensor, attn_stick],
            back_prop=False, parallel_iterations=1)

        kword_target_tensor = kword_target_tensor.stack()
        kword_target_tensor.set_shape([self.model_config.max_kword_len, self.model_config.batch_size])
        kword_target_tensor = tf.transpose(kword_target_tensor, perm=[1, 0])
        return tf.constant(10.0), kword_target_tensor, attn_stick

    def create_model(self):
        with tf.variable_scope('variables'):
            abstr_ph = []
            for _ in range(self.model_config.max_abstr_len):
                abstr_ph.append(tf.zeros(self.model_config.batch_size, tf.int32, name='abstract_input'))

            kwords_ph = []
            for _ in range(self.model_config.max_cnt_kword):
                kword = []
                for _ in range(self.model_config.max_kword_len):
                    kword.append(tf.zeros(self.model_config.batch_size, tf.int32, name='kword_input'))
                kwords_ph.append(kword)

            emb_abstr, emb_kword, proj_w, proj_b = self.get_embedding()
            abstr = tf.stack(self.embedding_fn(abstr_ph, emb_abstr), axis=1)
            kwords = []
            for kword_idx in range(self.model_config.max_cnt_kword):
                kwords.append(self.embedding_fn(kwords_ph[kword_idx], emb_kword))

        with tf.variable_scope('model_encoder'):
            if self.hparams.pos == 'timing':
                abstr = common_attention.add_timing_signal_1d(abstr)
            encoder_embed_inputs = tf.nn.dropout(abstr,
                                                 1.0 - self.hparams.layer_prepostprocess_dropout)
            abstr_bias = common_attention.attention_bias_ignore_padding(
                tf.to_float(tf.equal(tf.stack(abstr_ph, axis=1),
                                     self.voc_kword.encode(constant.SYMBOL_PAD))))
            abstr_outputs = transformer.transformer_encoder(
                encoder_embed_inputs, abstr_bias, self.hparams)

            if self.model_config.cov_mode == 'stick':
                attn_stick = tf.ones(
                    [self.model_config.batch_size, self.model_config.max_abstr_len, self.model_config.num_heads],
                    tf.float32, 'attn_stick')
            elif self.model_config.cov_mode == 'tuzhaopeng':
                attn_stick = tf.ones(
                    [self.model_config.batch_size, self.model_config.num_heads, 1,
                     self.model_config.dimension / self.model_config.num_heads],
                    tf.float32, 'attn_memory')

        losses = []
        targets = []
        obj = {}
        with tf.variable_scope('model_decoder'):
            for kword_idx in range(self.model_config.max_cnt_kword):
                if self.is_train:
                    kword = kwords[kword_idx][:-1]
                    kword_ph = kwords_ph[kword_idx]
                    kword_output_list, new_attn_stick = self.decode_step(kword, abstr_outputs, abstr_bias, attn_stick)
                    kword_logit_list = [self.output_to_logit(o, proj_w, proj_b) for o in kword_output_list]
                    kword_target_list = [tf.argmax(o, output_type=tf.int32, axis=-1)
                                           for o in kword_logit_list]
                    attn_stick = new_attn_stick

                    if self.model_config.number_samples > 0:
                        loss_fn = tf.nn.sampled_softmax_loss
                    else:
                        loss_fn = None
                    kword_lossbias = [
                        tf.to_float(tf.not_equal(d, self.voc_kword.encode(constant.SYMBOL_PAD)))
                        for d in kword_ph]
                    kword_lossbias = tf.stack(kword_lossbias, axis=1)
                    loss = sequence_loss(logits=tf.stack(kword_logit_list, axis=1),
                                         targets=tf.stack(kword_ph, axis=1),
                                         weights=kword_lossbias,
                                         softmax_loss_function=loss_fn,
                                         w=proj_w,
                                         b=proj_b,
                                         decoder_outputs=tf.stack(kword_output_list, axis=1),
                                         number_samples=self.model_config.number_samples
                                         )
                    targets.append(tf.stack(kword_target_list, axis=1))

                    if self.model_config.cov_mode == 'tuzhaopeng':
                        target_emb = tf.stack(self.embedding_fn(kword_target_list, emb_kword), axis=1)
                        target_emb = common_attention.split_heads(target_emb, self.model_config.num_heads)
                        target_emb = tf.reduce_mean(target_emb, axis=2)
                        target_emb_trans = tf.get_variable(
                            'dim_weight_trans',
                            shape=[1, target_emb.get_shape()[-1].value, target_emb.get_shape()[-1].value],
                            dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                        target_emb = tf.nn.conv1d(target_emb, target_emb_trans, 1, 'SAME')
                        target_emb = tf.expand_dims(target_emb, axis=2)
                        attn_stick += target_emb
                    losses.append(loss)
                else:
                    loss, target, new_attn_stick = self.transformer_beam_search(
                        abstr_outputs, abstr_bias, emb_kword, proj_w, proj_b,
                        attn_stick=attn_stick)
                    # loss, target, new_attn_stick = self.greed_search(kword_idx,
                    #     abstr_outputs, abstr_bias, emb_kword, proj_w, proj_b,
                    #     attn_stick=attn_stick)
                    targets.append(target)
                    losses = loss
                    attn_stick = new_attn_stick
                    if self.model_config.cov_mode == 'tuzhaopeng':
                        target.set_shape([self.model_config.batch_size, self.model_config.max_kword_len])
                        target_list = tf.unstack(target, axis=1)
                        target_emb = tf.stack(self.embedding_fn(target_list, emb_kword), axis=1)
                        target_emb = common_attention.split_heads(target_emb, self.model_config.num_heads)
                        target_emb = tf.reduce_mean(target_emb, axis=2)
                        target_emb_trans = tf.get_variable(
                            'dim_weight_trans',
                            shape=[1, target_emb.get_shape()[-1].value, target_emb.get_shape()[-1].value],
                            dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                        target_emb = tf.nn.conv1d(target_emb, target_emb_trans, 1, 'SAME')
                        target_emb = tf.expand_dims(target_emb, axis=2)
                        attn_stick += target_emb
                tf.get_variable_scope().reuse_variables()
        if targets:
            obj['targets'] = tf.stack(targets, axis=1)
        obj['abstr_ph'] = abstr_ph
        obj['kwords_ph'] = kwords_ph
        obj['attn_stick'] = attn_stick
        if type(losses) is list:
            losses = tf.add_n(losses)
        return losses, obj

    def create_model_multigpu(self):
        losses = []
        grads = []
        optim = self.get_optim()
        self.objs = []

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for gpu_id in range(self.model_config.num_gpus):
                with tf.device('/gpu:%d' % gpu_id):
                    loss, obj = self.create_model()
                    grad = optim.compute_gradients(loss)
                    losses.append(loss)
                    grads.append(grad)
                    self.objs.append(obj)
                    tf.get_variable_scope().reuse_variables()

        self.global_step = tf.get_variable(
            'global_step', initializer=tf.constant(0, dtype=tf.int64), trainable=False)
        with tf.variable_scope('optimization'):
            self.loss = tf.divide(tf.add_n(losses), self.model_config.num_gpus)
            self.perplexity = tf.exp(tf.reduce_mean(self.loss) / self.model_config.max_cnt_kword)

            if self.is_train:
                avg_grad = self.average_gradients(grads)
                grads = [g for (g,v) in avg_grad]
                clipped_grads, _ = tf.clip_by_global_norm(grads, self.model_config.max_grad_norm)
                self.train_op = optim.apply_gradients(zip(clipped_grads, tf.trainable_variables()), global_step=self.global_step)
                self.increment_global_step = tf.assign_add(self.global_step, 1)

            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

    def get_optim(self):
        learning_rate = tf.constant(self.model_config.learning_rate)

        if self.model_config.optimizer == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate)
        # Adam need lower learning rate
        elif self.model_config.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate)
        else:
            raise Exception('Not Implemented Optimizer!')
        return opt

    # Got from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101
    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def setup_hparams(self):
        self.hparams.num_heads = self.model_config.num_heads
        self.hparams.num_hidden_layers = self.model_config.num_hidden_layers
        self.hparams.num_encoder_layers = self.model_config.num_encoder_layers
        self.hparams.num_decoder_layers = self.model_config.num_decoder_layers
        self.hparams.pos = self.model_config.hparams_pos
        self.hparams.hidden_size = self.model_config.dimension
        self.hparams.layer_prepostprocess_dropout = self.model_config.layer_prepostprocess_dropout
        self.hparams.cov_mode = self.model_config.cov_mode

        if self.is_train:
            self.hparams.add_hparam('mode', tf.estimator.ModeKeys.TRAIN)
        else:
            self.hparams.add_hparam('mode', tf.estimator.ModeKeys.EVAL)
            self.hparams.layer_prepostprocess_dropout = 0.0
            self.hparams.attention_dropout = 0.0
            self.hparams.dropout = 0.0
            self.hparams.relu_dropout = 0.0