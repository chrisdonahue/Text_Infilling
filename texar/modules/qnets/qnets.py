from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from texar.module_base import ModuleBase
from texar.modules.networks.network_base import FeedForwardNetworkBase
from texar.agents.agent_utils import Space
from texar.utils import utils

__all__ = [
    'QNetBase',
    'CategoricalQNet'
]


class QNetBase(ModuleBase):
    def __init__(self,
                 network=None,
                 network_kwargs=None,
                 hparams=None):
        ModuleBase.__init__(self, hparams=hparams)

        with tf.variable_scope(self.variable_scope):
            self._build_network(network, network_kwargs)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        TODO
        """
        return {
            'network_type': 'FeedForwardNetwork',
            'network_hparams': {
                'layers': [
                    {'type': 'Dense',
                     'kwargs': {'units': 256, 'activation': 'relu'}},
                    {'type': 'Dense',
                     'kwargs': {'units': 256, 'activation': 'relu'}},
                ]
            },
            'distribution_kwargs': None,
            'name': 'policy_net',
            '@no_typecheck': ['network_type', 'network_hparams']
        }

    def _build_network(self, network, kwargs):
        if network is not None:
            self._network = network
        else:
            kwargs = utils.get_instance_kwargs(
                kwargs, self._hparams.network_hparams)
            self._network = utils.check_or_get_instance(
                self._hparams.network_type,
                kwargs,
                module_paths=['texar.modules', 'texar.custom'],
                classtype=FeedForwardNetworkBase)

    def _build(self, inputs, mode=None): # pylint: disable=arguments-differ
        raise NotImplementedError

    @property
    def network(self):
        """The network.
        """
        return self._network


class CategoricalQNet(QNetBase):
    def __init__(self,
                 action_space=None,
                 network=None,
                 network_kwargs=None,
                 hparams=None):
        QNetBase.__init__(self, hparams=hparams)

        with tf.variable_scope(self.variable_scope):
            if action_space is None:
                action_space = Space(
                    low=0, high=self._hparams.action_space, dtype=np.int32)
            self._action_space = action_space
            self._append_output_layer()

    @staticmethod
    def default_hparams():
        hparams = QNetBase.default_hparams()
        hparams.update({
            'action_space': 2,
            'make_output_layer': True})
        return hparams

    def _append_output_layer(self):
        if not self._hparams.make_output_layer:
            return

        if self._action_space.shape != ():
            raise ValueError('Only scalar discrete action is supported.')
        else:
            output_size = self._action_space.high - self._action_space.low

        layer_hparams = {
            'type': 'Dense',
            'kwargs': {'units': output_size}}
        self._network.append_layer(layer_hparams)

    def _build(self, inputs, mode=None):
        outputs = dict(qvalues=self._network(inputs, mode=mode))

        if not self._built:
            self._add_internal_trainable_variables()
            self._add_trainable_variable(self._network.trainable_variables)
            self._built = True

        return outputs

    @property
    def action_space(self):
        """An instance of :class:`~texar.agents.Space` specifiying the
        action space.
        """
        return self._action_space