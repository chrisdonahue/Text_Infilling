#
"""
Various embedders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.modules.embedders.embedder_base import EmbedderBase
from texar.modules.embedders import embedder_utils
from texar.utils import utils

__all__ = [
    "WordEmbedder"
]

#TODO(zhiting): add soft-embedder, position-embedder, embedder combiner


class WordEmbedder(EmbedderBase):
    """Simple word embedder that maps indexes into embeddings via lookup.

    Either :attr:`init_value` or :attr:`vocab_size` is required. If both are
    given, :attr:`init_value.shape[0]` must equal :attr:`vocab_size`.

    Args:
        init_value (optional): A `Tensor` or numpy array that contains the
            initial value of embeddings. It is typically of shape
            `[vocab_size, embedding dim]`

            If `None`, embedding is initialized as specified in
            :attr:`hparams["initializer"]`. Otherwise, the
            :attr:`"initializer"` and :attr:`"dim"`
            hyperparameters in :attr:`hparams` are ignored.
        vocab_size (int, optional): The vocabulary size. Required if
            :attr:`init_value` is not given.
        hparams (dict, optional): Embedder hyperparameters. If it is not
            specified, the default hyperparameter setting is used. See
            :attr:`default_hparams` for the sturcture and default values.
    """

    def __init__(self, init_value=None, vocab_size=None, hparams=None):
        EmbedderBase.__init__(self, hparams=hparams)

        if init_value is None and vocab_size is None:
            raise ValueError(
                "Either `init_value` or `vocab_size` is required.")

        self._init_parameterized_embedding(init_value, vocab_size,
                                           self._hparams)

        self._vocab_size = vocab_size
        if vocab_size is None:
            self._vocab_size = self._num_embeds
        if self._vocab_size != self._num_embeds:
            raise ValueError(
                'vocab_size must equal to init_value.shape[0].'
                'Got %d and %d' % (self._vocab_size, self._num_embeds))

        self._built = True

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            A dictionary with the following structure and values.

            .. code-block:: python

                {
                    "name": "word_embedder",
                    "dim": 100,
                    "initializer": {
                        "type": "random_uniform_initializer",
                        "kwargs": {
                            "minval": -0.1,
                            "maxval": 0.1,
                            "seed": None
                        }
                    },
                    "regularizer": {
                        "type": "L1L2",
                        "kwargs": {
                            "l1": 0.,
                            "l2": 0.
                        }
                    },
                    "dropout_rate": 0,
                    "trainable": True,
                }

            See :func:`~texar.modules.default_embedding_hparams` for more
            details.
        """
        hparams = embedder_utils.default_embedding_hparams()
        hparams["name"] = "word_embedder"
        return hparams

    def _build(self, inputs, mode=None, **kwargs):
        """Embeds inputs with look-up.

        Args:
            inputs: An integer tensor containing the ids to be looked up.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
                `TRAIN`, `EVAL`, and `PREDICT`. If `None`, dropout will be
                controlled by :func:`texar.context.global_mode`.
            kwargs: Additional keyword arguments for
                :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>` besides
                :attr:`params` and :attr:`ids`.

        Returns:
            A `Tensor` of shape `shape(inputs) + embedding dimension`.
        """
        embedding = self._embedding
        if self._dropout_layer is not None:
            is_training = utils.is_train_mode(mode)
            embedding = self._dropout_layer.apply(
                inputs=embedding, training=is_training)
        outputs = tf.nn.embedding_lookup(embedding, inputs, **kwargs)
        return outputs

    @property
    def embedding(self):
        """The embedding tensor.
        """
        return self._embedding

    @property
    def dim(self):
        """The embedding dimension.
        """
        return self._dim

    @property
    def vocab_size(self):
        """The vocabulary size.
        """
        return self._vocab_size

