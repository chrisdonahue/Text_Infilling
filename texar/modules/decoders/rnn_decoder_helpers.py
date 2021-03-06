#
"""
Various helper classes and utilities for RNN decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.seq2seq import TrainingHelper as TFTrainingHelper
from tensorflow.contrib.seq2seq import Helper as TFHelper
from tensorflow.contrib.distributions import RelaxedOneHotCategorical \
    as GumbelSoftmax

from texar.utils import utils

# pylint: disable=not-context-manager, too-many-arguments
# pylint: disable=too-many-instance-attributes

__all__ = [
    "default_helper_train_hparams",
    "default_helper_infer_hparams",
    "get_helper",
    "_get_training_helper",
    "GumbelSoftmaxEmbeddingHelper",
    "SoftmaxEmbeddingHelper",
]

def default_helper_train_hparams():
    """Returns default hyperparameters of an RNN decoder helper in the training
    phase.

    See also :meth:`~texar.modules.decoders.rnn_decoder_helpers.get_helper`
    for information of the hyperparameters.

    Returns:
        dict: A dictionary with following structure and values:

        .. code-block:: python

            {
                # The `helper_type` argument for `get_helper`, i.e., the name
                # or full path to the helper class.
                "type": "TrainingHelper",

                # The `**kwargs` argument for `get_helper`, i.e., additional
                # keyword arguments for constructing the helper.
                "kwargs": {}
            }
    """
    return {
        "type": "TrainingHelper",
        "kwargs": {}
    }

def default_helper_infer_hparams():
    """Returns default hyperparameters of an RNN decoder helper in the inference
    phase.

    See also :meth:`~texar.modules.decoders.rnn_decoder_helpers.get_helper`
    for information of the hyperparameters.

    Returns:
        dict: A dictionary with following structure and values:

        .. code-block:: python

            {
                # The `helper_type` argument for `get_helper`, i.e., the name
                # or full path to the helper class.
                "type": "SampleEmbeddingHelper",

                # The `**kwargs` argument for `get_helper`, i.e., additional
                # keyword arguments for constructing the helper.
                "kwargs": {}
            }
    """
    return {
        "type": "SampleEmbeddingHelper",
        "kwargs": {}
    }


def get_helper(helper_type,
               inputs=None,
               sequence_length=None,
               embedding=None,
               start_tokens=None,
               end_token=None,
               **kwargs):
    """Creates a Helper instance.

    Args:
        helper_type (str): The name or full path to the helper class.
            E.g., the classname of the built-in helpers in
            :mod:`texar.modules.decoders.rnn_decoder_helpers` or
            :mod:`tensorflow.contrib.seq2seq`, or the classname of user-defined
            helpers in :mod:`texar.custom`, or a full path like
            "my_module.MyHelper".
        inputs ((structure of) Tensors, optional): Inputs to the decoder.
        sequence_length (1D integer array or Tensor, optional): Lengths of input
            token sequences.
        embedding (optional): A callable that takes a vector tensor of integer
            indexes, or the `params` argument for `embedding_lookup` (e.g.,
            the embedding Tensor).
        start_tokens (int array or 1D int Tensor, optional): Of shape
            `[batch_size]`. The start tokens.
        end_token (int or int scalar Tensor, optional): The token that marks
            end of decoding.
        **kwargs: additional keyword arguments for constructing the helper.

    Returns:
        An instance of specified helper.
    """
    module_paths = [
        'texar.modules.decoders.rnn_decoder_helpers',
        'tensorflow.contrib.seq2seq',
        'texar.custom']
    class_kwargs = {"inputs": inputs,
                    "sequence_length": sequence_length,
                    "embedding": embedding,
                    "start_tokens": start_tokens,
                    "end_token": end_token}
    class_kwargs.update(kwargs)
    return utils.get_instance_with_redundant_kwargs(
        helper_type, class_kwargs, module_paths)


def _get_training_helper( #pylint: disable=invalid-name
        inputs, sequence_length, embedding=None, time_major=False, name=None):
    """Returns an instance of :tf_main:`TrainingHelper
    <contrib/seq2seq/TrainingHelper>` given embeddings.

    Args:
        inputs: If :attr:`embedding` is given, this is sequences of input
            token indexes. If :attr:`embedding` is `None`, this is passed to
            TrainingHelper directly.
        sequence_length (1D Tensor): Lengths of input token sequences.
        embedding (optional): The `params` argument of
        :tf_main:`tf.nn.embedding_lookup
        <nn/embedding_lookup>` (e.g., the embedding Tensor); or a callable that
        takes a vector of integer indexes and returns respective embedding.
        time_major (bool): Whether the tensors in `inputs` are time major.
            If `False` (default), they are assumed to be batch major.
        name (str, optional): Name scope for any created operations.

    Returns:
        An instance of TrainingHelper.

    Raises:
        ValueError: if `sequence_length` is not a 1D tensor.
    """
    if embedding is None:
        return TFTrainingHelper(inputs=inputs,
                                sequence_length=sequence_length,
                                time_major=time_major,
                                name=name)

    with tf.name_scope(name, "TrainingHelper", [embedding, inputs]):
        if callable(embedding):
            embedding_fn = embedding
        else:
            embedding_fn = (
                lambda ids: tf.nn.embedding_lookup(embedding, ids))
        emb_inputs = embedding_fn(inputs)
    helper = TFTrainingHelper(inputs=emb_inputs,
                              sequence_length=sequence_length,
                              time_major=time_major,
                              name=name)
    return helper


class SoftmaxEmbeddingHelper(TFHelper):
    """A helper that feed softmax to the next step.

    Use the softmax probability to pass through an embedding layer to get the
    next input.
    """

    def __init__(self, embedding, start_tokens, end_token, tau,
                 stop_gradient=False, use_finish=True):
        """Initializer.

        Args:
            embedding: An embedding argument (:attr:`params`) for
                :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>`. Note
                that a callable is not acceptable here.
            start_tokens: An `int32` vector tensor  shaped `[batch_size]`. The
                start tokens.
            end_token: An `int32` scalar tensor. The token that marks end of
                decoding.
            tau: softmax anneal temperature.
            stop_gradient: stop the gradient when feeding to the next step.
        """

        if callable(embedding):
            raise ValueError("embedding must be an embedding matrix.")
        else:
            self._embedding = embedding
            self._embedding_fn = (
                lambda ids: tf.nn.embedding_lookup(embedding, ids))

        self._start_tokens = tf.convert_to_tensor(
            start_tokens, dtype=tf.int32, name="start_tokens")
        self._end_token = tf.convert_to_tensor(
            end_token, dtype=tf.int32, name="end_token")
        self._start_inputs = self._embedding_fn(self._start_tokens)
        self._batch_size = tf.size(self._start_tokens)
        self._tau = tau
        self._stop_gradient = stop_gradient
        self._use_finish = use_finish

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_dtype(self):
        return tf.float32

    @property
    def sample_ids_shape(self):
        return self._embedding.get_shape()[:1]

    def initialize(self, name=None):
        finished = tf.tile([False], [self._batch_size])
        return (finished, self._start_inputs)

    def sample(self, time, outputs, state, name=None):
        sample_ids = tf.nn.softmax(outputs / self._tau)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        if self._use_finish:
            hard_ids = tf.argmax(sample_ids, axis=-1, output_type=tf.int32)
            finished = tf.equal(hard_ids, self._end_token)
        else:
            finished = tf.tile([False], [self._batch_size])
        if self._stop_gradient:
            sample_ids = tf.stop_gradient(sample_ids)
        next_inputs = tf.matmul(sample_ids, self._embedding)
        return (finished, next_inputs, state)


class GumbelSoftmaxEmbeddingHelper(SoftmaxEmbeddingHelper):
    """A helper that use Gumbel Softmax sampling.

    Use the Gumbel Softmax sample and pass the sample through an embedding
    layer to get the next input.
    """

    def __init__(self, embedding, start_tokens, end_token, tau,
                 straight_through=False, stop_gradient=False, use_finish=True):
        """Initializer.

        Args:
            embedding: An embedding argument (:attr:`params`) for
                :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>`. Note
                that a callable is not acceptable here.
            start_tokens: An `int32` vector tensor shaped `[batch_size]`. The
                start tokens.
            end_token: An `int32` scalar tensor. The token that marks end of
                decoding.
            tau: anneal temperature for sampling.
            straight_through: whether to use the straight through estimator.
            stop_gradient: stop gradients when feeding to the next step.
        """
        super(GumbelSoftmaxEmbeddingHelper, self).__init__(
            embedding, start_tokens, end_token, tau, stop_gradient, use_finish)
        self._straight_through = straight_through

    def sample(self, time, outputs, state, name=None):
        sample_ids = GumbelSoftmax(self._tau, logits=outputs).sample()
        if self._straight_through:
            size = tf.shape(sample_ids)[-1]
            sample_ids_hard = tf.cast(
                tf.one_hot(tf.argmax(sample_ids, -1), size), sample_ids.dtype)
            sample_ids = tf.stop_gradient(sample_ids_hard - sample_ids) \
                         + sample_ids
        return sample_ids
