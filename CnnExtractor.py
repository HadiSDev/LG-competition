import numpy as np
from stable_baselines.common.policies import CnnPolicy, FeedForwardPolicy, ActorCriticPolicy
import tensorflow as tf
from stable_baselines.common.tf_layers import conv, conv_to_fc, linear


class CnnExtractor(FeedForwardPolicy):

    def __init__(self, *args, **kwargs,):
        super(CnnExtractor, self).__init__(*args, **kwargs,
                                           cnn_extractor=custom_cnn,
                                           act_fun=tf.nn.relu,
                                           net_arch=[32, dict(pi=[64, 64], vf=[64, 64])],
                                           feature_extraction='cnn')


def custom_cnn(scaled_images, **kwargs):
    """

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=18, filter_size=2, stride=1, **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=2, stride=1, **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=2, stride=1, **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=64))


print(issubclass(CnnExtractor, ActorCriticPolicy))
