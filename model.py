# Copyright 2019-2022 by Andrey Ignatov. All Rights Reserved.

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def PyNET(input, instance_norm=False, instance_norm_level_1=False):
    k = 2
    with tf.name_scope("generator_3_"):

        # -----------------------------------------
        # Downsampling layers
        x = input

        conv_l1_d1 =  _conv_layer(x, 32, 3, 1, relu=True, instance_norm=False, padding='SAME')
        conv_l2_d1 = _conv_layer(conv_l1_d1, 64, 2, 2, relu=True, instance_norm=False, padding='VALID')
        conv_l3_d1 = _conv_layer(conv_l2_d1, 128, 2, 2, relu=True, instance_norm=False, padding='VALID')

        conv_l3_d6 = _residual_groups_residual(conv_l3_d1, 128, instance_norm=True, groups=4, n=2)
        conv_l3_d8 = _sam_block(conv_l3_d6, 128, instance_norm=False) + conv_l3_d6
        conv_l3_d9 = _cam_block(conv_l3_d8, 128, instance_norm=False) + conv_l3_d8

        # -> Output: Level 3

        conv_l3_out = _conv_layer(conv_l3_d9, 3 * k * k, 1, 1, relu=False, instance_norm=False)  # 32 -> 128  # 128ch -> 48ch
        conv_l3_out = tf.nn.depth_to_space(conv_l3_out, k, data_format='NHWC', name='depth_to_space_3')
        output_l3 = tf.nn.tanh(conv_l3_out) * 0.58 + 0.5

    with tf.name_scope("generator_2_"):
        conv_t2a = _upsample_layer(conv_l3_d9, 64, 2, 2)
        conv_l2_d2 = _conv_layer(conv_l2_d1, 64, 3, 1, relu=True, instance_norm=False, padding='SAME')

        conv_l2_d3 = conv_l2_d2 + conv_t2a
                
        conv_l2_d12 = _conv_residual_1x1(conv_l2_d3, 64, instance_norm=False)
        conv_l2_d13 = _residual_groups_residual(conv_l2_d12, 64, instance_norm=True, groups=2, n=3)
        conv_l2_d16 = _cam_block(conv_l2_d13, 64, instance_norm=False) + conv_l2_d13

        # -> Output: Level 2

        conv_l2_out = _conv_layer(conv_l2_d16, 3 * k * k, 1, 1, relu=False, instance_norm=False)
        conv_l2_out = tf.nn.depth_to_space(conv_l2_out, k, data_format='NHWC', name='depth_to_space_2')
        output_l2 = tf.nn.tanh(conv_l2_out) * 0.58 + 0.5

    with tf.name_scope("generator_1_"):
        conv_t1a = _upsample_layer(conv_l2_d16, 32, 2, 2)
        conv_l1_d2 = _conv_layer(conv_l1_d1, 32, 3, 1, relu=True, instance_norm=False, padding='SAME')

        conv_l1_d3 = conv_l1_d2 + conv_t1a

        conv_l1_d12 = _conv_residual_1x1(conv_l1_d3, 32, instance_norm=False)
        conv_l1_d13 = _residual_groups_residual(conv_l1_d12, 32, instance_norm=True, groups=4)
        conv_l1_d14 = _residual_groups_residual(conv_l1_d13, 32, instance_norm=True, groups=2)

        # -> Output: Level 1
        conv_l1_out = _conv_layer(conv_l1_d14, 4 * k * k, 1, 1, relu=True, instance_norm=False)
        conv_l1_out = tf.nn.depth_to_space(conv_l1_out, k, data_format='NHWC', name='depth_to_space_1')
        conv_l1_out = _conv_layer(conv_l1_out, 3, 3, 1, relu=False, instance_norm=False)
        output_l1 = tf.nn.tanh(conv_l1_out) * 0.58 + 0.5

    with tf.name_scope("generator_0_"):
        conv_l0 = _upsample_layer(conv_l1_d14, 32, 3, 2)
        conv_l0_out = _conv_layer(conv_l0, 3 * k * k, 3, 1, relu=False, instance_norm=False)

        # -> Output: Level 0
        output_l0 = tf.nn.tanh(conv_l0_out) * 0.58 + 0.5

    return None, output_l1, output_l2, output_l3


def _conv_residual_1x1(input, num_maps, instance_norm):

    conv_3a = _conv_layer(input, num_maps, 1, 1, relu=False, instance_norm=instance_norm)

    output_tensor = tf.keras.layers.PReLU(shared_axes=[1,2])(conv_3a) + input

    return output_tensor


def _residual_groups_residual(input, num_maps, instance_norm, groups=4, n=1):

    groups_tf = []
    
    batch, rows, cols, channels = [i for i in input.get_shape()]

    step = int(channels) // groups
    assert(int(channels) % groups == 0)

    for i in range(groups):
        groups_tf.append(input[:, :, :, step*i:step*(i+1)])

    values = []
    for i, g in enumerate(groups_tf):
        for k in range(n):
            g = _conv_layer(g, num_maps // groups, 3, 1, relu=True, instance_norm=(instance_norm and (i % 2) == 0)) + g
        values.append(g)

    conv_3a = tf.concat(values, axis=-1)

    conv_3a = _conv_layer(conv_3a, num_maps, 1, 1, relu=False, instance_norm=False)

    output_tensor = conv_3a + input

    return output_tensor


def _sam_block(input, num_maps, instance_norm):
    out = _conv_layer(input, num_maps, 3, 1, relu=True, instance_norm=instance_norm)

    spatial_att = _dw_conv_layer(out, num_maps, 5, 1, relu=False, instance_norm=False)
    spatial_att = tf.math.sigmoid(spatial_att)

    return spatial_att * out


def _cam_block(input, num_maps, instance_norm=False):

    out = _conv_layer(input, num_maps, 3, 1, relu=True, instance_norm=instance_norm)

    channel_att = _conv_layer(out, num_maps, 1, 3, relu=True, instance_norm=False, padding='VALID')
    channel_att = _conv_layer(channel_att, num_maps, 3, 3, relu=True, instance_norm=False, padding='VALID')
    channel_att = tf.math.reduce_mean(channel_att, axis=[1,2], keepdims=True, name=None)
    channel_att = _conv_layer(channel_att, num_maps, 1, 1, relu=True, instance_norm=False, padding='VALID')
    channel_att = _conv_layer(channel_att, num_maps, 1, 1, relu=False, instance_norm=False, padding='VALID')
    channel_att = tf.math.sigmoid(channel_att)

    return  channel_att * out


def stack(x, y):
    return tf.concat([x, y], -1)


def _dw_conv_layer(net, num_filters, filter_size, strides, relu=True, instance_norm=False, padding='SAME'):

    if filter_size // 2 >= 1:
        paddings = tf.constant([[0, 0], [filter_size // 2, filter_size // 2], [filter_size // 2, filter_size // 2], [0, 0]])
        net = tf.pad(net, paddings, mode='REFLECT')
    
    net =  tf.keras.layers.DepthwiseConv2D(
        filter_size, strides=(strides, strides), padding='valid', depth_multiplier=1,
        data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True,
        depthwise_initializer='glorot_uniform',
        bias_initializer='zeros', depthwise_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None,
        bias_constraint=None
    )(net)

    if instance_norm:
        net = _instance_norm(net)

    if relu:
        net = tf.keras.layers.PReLU(shared_axes=[1,2])(net)

    return net


def _conv_layer(net, num_filters, filter_size, strides, relu=True, instance_norm=False, padding='SAME'):

    if filter_size // 2 >= 1 and padding == 'SAME':
        paddings = tf.constant([[0, 0], [filter_size // 2, filter_size // 2], [filter_size // 2, filter_size // 2], [0, 0]])
        net = tf.pad(net, paddings, mode='REFLECT')
    
    net = tf.keras.layers.Conv2D(
        num_filters, filter_size, strides=(strides, strides), padding='valid',
        data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
        use_bias=True, kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=1),
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None)(net)

    if instance_norm:
        net = _instance_norm(net)

    if relu:
        net = tf.keras.layers.PReLU(shared_axes=[1,2])(net)

    return net


def _instance_norm(net):
    return tfa.layers.InstanceNormalization()(net)


def _conv_tranpose_layer(net, num_filters, filter_size, strides, relu=True):

    net = tf.keras.layers.Conv2DTranspose(
        num_filters, filter_size, strides=(strides, strides), padding='same',
        output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None,
        use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None
    )(net)
    
    if relu:
        return tf.keras.layers.PReLU(shared_axes=[1,2])(net)
    else:
        return net


def _upsample_layer(net, num_filters, filter_size, strides, relu=True):
    net = tf.keras.layers.UpSampling2D(
        size=(strides, strides), data_format=None, interpolation='bilinear',
    )(net)

    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    net = tf.pad(net, paddings, mode='REFLECT')
    
    net = tf.keras.layers.Conv2D(
        num_filters, 3, strides=(1, 1), padding='valid',
        data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
        use_bias=True, kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=1),
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None)(net)
    
    if relu:
        return tf.keras.layers.PReLU(shared_axes=[1,2])(net)
    else:
        return net


def max_pool(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

