"""
A RNN-Based Discriminator for SeqGAN.
"""
import tensorflow as tf
import texar as tx
from texar.modules.classifiers.conv_classifiers import Conv1DClassifier
from utils import *


class Discriminator:
    def __init__(self, config, word2id, class_num=2):
        initializer = tf.random_uniform_initializer(
            -config.init_scale, config.init_scale)
        with tf.variable_scope('discriminator', initializer=initializer):
            self.batch_size = config.batch_size
            self.max_seq_length = config.num_steps
            self.vocab_size = len(word2id)
            self.class_num = class_num

            self.labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, 1], name="labels")
            self.samples = tf.placeholder(dtype=tf.int32, name="samples",
                                          shape=[self.batch_size, self.max_seq_length + 1])

            self.classifier = Conv1DClassifier()
            self.embedder = tx.modules.WordEmbedder(
                vocab_size=self.vocab_size, hparams=config.emb)
            emb_inputs = self.embedder(self.samples)
            self.ypred_for_auc, self.predictions = self.classifier(emb_inputs)

            # Calculate loss
            self.mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=self.labels,  # [batch, 1]
                logits=self.ypred_for_auc[:, tf.newaxis],  # [batch, 1, num_class]
                sequence_length=[1] * self.batch_size)

            self.global_step = tf.placeholder(tf.int32)
            self.train_op = tx.core.get_train_op(
                self.mle_loss, global_step=self.global_step, increment_global_step=False,
                hparams=config.opt)