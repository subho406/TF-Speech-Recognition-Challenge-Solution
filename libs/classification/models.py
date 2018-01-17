# Cognitbit Solutions LLP
#
# Author: Subhojeet Pramanik
# ==============================================================================
"""Model definitions for simple speech recognition.
    create_conv_model: Basic Convolutional Model
    
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from . import resnet_model
from .inception.slim import inception_model


def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count,use_spectrogram=False):
    """Calculates common settings needed for all models.
  
    Args:
      label_count: How many classes are to be recognized.
      sample_rate: Number of audio samples per second.
      clip_duration_ms: Length of each audio clip to be analyzed.
      window_size_ms: Duration of frequency analysis window.
      window_stride_ms: How far to move in time between frequency windows.
      dct_coefficient_count: Number of frequency bins to use for analysis.
  
    Returns:
      Dictionary containing common settings.
    """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = dct_coefficient_count * spectrogram_length
    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'dct_coefficient_count': dct_coefficient_count,
        'fingerprint_size': fingerprint_size,
        'label_count': label_count,
        'sample_rate': sample_rate,
    }


def create_model(fingerprint_input, model_settings, model_architecture,
                 is_training, runtime_settings=None,model_size_info=None):
    """Builds a model of the requested architecture compatible with the settings.
  
    There are many possible ways of deriving predictions from a spectrogram
    input, so this function provides an abstract interface for creating different
    kinds of models in a black-box way. You need to pass in a TensorFlow node as
    the 'fingerprint' input, and this should output a batch of 1D features that
    describe the audio. Typically this will be derived from a spectrogram that's
    been run through an MFCC, but in theory it can be any feature vector of the
    size specified in model_settings['fingerprint_size'].
  
    The function will build the graph it needs in the current TensorFlow graph,
    and return the tensorflow output that will contain the 'logits' input to the
    softmax prediction process. If training flag is on, it will also return a
    placeholder node that can be used to control the dropout amount.
  
    See the implementations below for the possible model architectures that can be
    requested.
  
    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      model_architecture: String specifying which kind of model to create.
      is_training: Whether the model is going to be used for training.
      runtime_settings: Dictionary of information about the runtime.
  
    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
  
    Raises:
      Exception: If the architecture type isn't recognized.
    """
    if model_architecture == 'single_fc':
        return create_single_fc_model(fingerprint_input, model_settings,
                                      is_training)
    elif model_architecture == 'conv':
        return create_conv_model(fingerprint_input, model_settings, is_training)
    elif model_architecture == 'low_latency_conv':
        return create_low_latency_conv_model(fingerprint_input, model_settings,
                                             is_training)
    elif model_architecture == 'low_latency_svdf':
        return create_low_latency_svdf_model(fingerprint_input, model_settings,
                                             is_training, runtime_settings)
    elif model_architecture == 'convlstm':
        return create_multilayer_convlstm_model(fingerprint_input, model_settings,
                                                is_training)
    elif model_architecture == 'lstm_l':
        return create_lstm_l_model(fingerprint_input,model_settings,is_training)
    elif model_architecture == 'ds_cnn':
        return create_ds_cnn_model(fingerprint_input,model_settings,model_size_info,is_training)
    elif model_architecture == 'ds_cnn_spec':
        return create_ds_cnn_model(fingerprint_input,model_settings,model_size_info,is_training)
    elif model_architecture == 'inception':
        return create_inception_model(fingerprint_input,model_settings,is_training)
    elif model_architecture == 'c_rnn':
        return create_crnn_model(fingerprint_input,model_settings,model_size_info,is_training)
    elif model_architecture=='c_rnn_spec':
        return create_crnn_model(fingerprint_input,model_settings,model_size_info,is_training)
    elif model_architecture=='gru':
        return create_gru_model(fingerprint_input,model_settings,model_size_info,is_training)
    else:
        raise Exception('model_architecture argument "' + model_architecture +
                        '" not recognized, should be one of "single_fc", "conv",' +
                        ' "low_latency_conv, or "low_latency_svdf"')


def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.
  
    Args:
      sess: TensorFlow session.
      start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)

    
def create_gru_model(fingerprint_input, model_settings, model_size_info, 
                       is_training):
  """Builds a model with multi-layer GRUs
  model_size_info: [number of GRU layers, number of GRU cells per layer]
  Optionally, the bi-directional GRUs and/or GRU with layer-normalization 
    can be explored.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size])

  num_classes = model_settings['label_count']

  layer_norm = False
  bidirectional = False

  num_layers = model_size_info[0]
  gru_units = model_size_info[1]

  gru_cell_fw = []
  gru_cell_bw = []
  if layer_norm:
    for i in range(num_layers):
      gru_cell_fw.append(LayerNormGRUCell(gru_units))
      if bidirectional:
        gru_cell_bw.append(LayerNormGRUCell(gru_units))
  else:
    for i in range(num_layers):
      gru_cell_fw.append(tf.contrib.rnn.GRUCell(gru_units))
      if bidirectional:
        gru_cell_bw.append(tf.contrib.rnn.GRUCell(gru_units))
  
  if bidirectional:
    outputs, output_state_fw, output_state_bw = \
      tf.contrib.rnn.stack_bidirectional_dynamic_rnn(gru_cell_fw, gru_cell_bw, 
      fingerprint_4d, dtype=tf.float32)
    flow = outputs[:, -1, :]
  else:
    cells = tf.contrib.rnn.MultiRNNCell(gru_cell_fw)
    _, last = tf.nn.dynamic_rnn(cell=cells, inputs=fingerprint_4d, 
                dtype=tf.float32)
    flow = last[-1]

  with tf.name_scope('Output-Layer'):
    W_o = tf.get_variable('W_o', shape=[flow.get_shape()[-1], num_classes], 
            initializer=tf.contrib.layers.xavier_initializer())
    b_o = tf.get_variable('b_o', shape=[num_classes])
    logits = tf.matmul(flow, W_o) + b_o

  if is_training:
    return logits, dropout_prob
  else:
    return logits


def create_inception_model(fingerprint_input,model_settings,is_training):
    """
    Create a Resnet Model
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
        train_mode_placeholder=tf.placeholder(tf.bool,name='train_mode')
    else:
        train_mode_placeholder=False
    batch_size = tf.shape(fingerprint_input)[0]
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
    paddings = tf.constant([[0, 0],[0,299-input_time_size],[0,299-input_frequency_size],[0,2]])
    fingerprint_4d=tf.pad(fingerprint_4d, paddings, "CONSTANT")

    label_count=model_settings['label_count']
    logits,_=inception_model.inception_v3(fingerprint_4d,
                 dropout_keep_prob=0.8,
                 num_classes=label_count,
                 is_training=True,
                 restore_logits=True,
                 scope='')
    if is_training:
        return logits,dropout_prob
    else:
        return logits

def create_single_fc_model(fingerprint_input, model_settings, is_training):
    """Builds a model with a single hidden fully-connected layer.
  
    This is a very simple model with just one matmul and bias layer. As you'd
    expect, it doesn't produce very accurate results, but it is very fast and
    simple, so it's useful for sanity testing.
  
    Here's the layout of the graph:
  
    (fingerprint_input)
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
  
    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.
  
    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']
    weights = tf.Variable(
        tf.truncated_normal([fingerprint_size, label_count], stddev=0.001))
    bias = tf.Variable(tf.zeros([label_count]))
    logits = tf.matmul(fingerprint_input, weights) + bias
    if is_training:
        return logits, dropout_prob
    else:
        return logits


def create_conv_model(fingerprint_input, model_settings, is_training):
    """Builds a standard convolutional model.
  
    This is roughly the network labeled as 'cnn-trad-fpool3' in the
    'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
    http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
  
    Here's the layout of the graph:
  
    (fingerprint_input)
            v
        [Conv2D]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
          [Relu]
            v
        [MaxPool]
            v
        [Conv2D]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
          [Relu]
            v
        [MaxPool]
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
  
    This produces fairly good quality results, but can involve a large number of
    weight parameters and computations. For a cheaper alternative from the same
    paper with slightly less accuracy, see 'low_latency_conv' below.
  
    During training, dropout nodes are introduced after each relu, controlled by a
    placeholder.
  
    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.
  
    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
    first_filter_width = 8
    first_filter_height = 20
    first_filter_count = 64
    first_weights = tf.Variable(
        tf.truncated_normal(
            [first_filter_height, first_filter_width, 1, first_filter_count],
            stddev=0.01))
    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                              'SAME') + first_bias
    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu
    max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    second_filter_width = 4
    second_filter_height = 10
    second_filter_count = 64
    second_weights = tf.Variable(
        tf.truncated_normal(
            [
                second_filter_height, second_filter_width, first_filter_count,
                second_filter_count
            ],
            stddev=0.01))
    second_bias = tf.Variable(tf.zeros([second_filter_count]))
    second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                               'SAME') + second_bias
    second_relu = tf.nn.relu(second_conv)
    if is_training:
        second_dropout = tf.nn.dropout(second_relu, dropout_prob)
    else:
        second_dropout = second_relu
    second_conv_shape = second_dropout.get_shape()
    second_conv_output_width = second_conv_shape[2]
    second_conv_output_height = second_conv_shape[1]
    second_conv_element_count = int(
        second_conv_output_width * second_conv_output_height *
        second_filter_count)
    flattened_second_conv = tf.reshape(second_dropout,
                                       [-1, second_conv_element_count])
    label_count = model_settings['label_count']
    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [second_conv_element_count, label_count], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def resCONVLSTM(inputs, model_settings, is_training, name=''):
    """Creates a Residual ConvLSTM as in https://arxiv.org/abs/1607.06450.
        1-D Conv on feature, unidirectional rnn
        
    """
    with(tf.variable_scope('resCONVLSTM_%s' % name)):
        batch_size = tf.shape(inputs)[0]
        input_frequency_size = model_settings['dct_coefficient_count']
        input_time_size = model_settings['spectrogram_length']
        input_shape = [input_frequency_size, 1]
        conv1 = tf.contrib.rnn.ConvLSTMCell(1, input_shape, 1, [10], name='conv1')
        conv2 = tf.contrib.rnn.ConvLSTMCell(1, input_shape, 1, [10], name='conv2')
        # First ConvLSTM
        initial_conv1 = conv1.zero_state(batch_size, dtype=tf.float32)
        initial_conv2 = conv2.zero_state(batch_size, dtype=tf.float32)
        conv1_o, _ = tf.nn.dynamic_rnn(conv1, inputs, initial_state=initial_conv1)
        bn1 = slim.layers.batch_norm(conv1_o, is_training=is_training,updates_collections=None,decay=0.96,
                                           zero_debias_moving_mean=True,data_format='NCHW')
        bn1_relu = tf.nn.relu(bn1)
        conv2_o, _ = tf.nn.dynamic_rnn(conv2, bn1_relu, initial_state=initial_conv2)
        bn2 = slim.layers.batch_norm(conv2_o, is_training=is_training,updates_collections=None,decay=0.96,
                                          zero_debias_moving_mean=True,activation_fn=tf.nn.relu,data_format='NCHW')
        residual = tf.add(bn2, inputs)
        output_relu = tf.nn.relu(residual)
        return output_relu


def create_multilayer_convlstm_model(fingerprint_input, model_settings, is_training):
    """
        Creates a Multilayer ConvLSTM Model Followed by a linear layer and softmax activation function



    """
    
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
        train_mode_placeholder=tf.placeholder(tf.bool,name='train_mode')
    else:
        train_mode_placeholder=False
    batch_size = tf.shape(fingerprint_input)[0]
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
    
    # Layer1 resCONVLSTMs
    resCONVLSTM1 = resCONVLSTM(fingerprint_4d, model_settings, train_mode_placeholder, '1')
    resCONVLSTM2 = resCONVLSTM(resCONVLSTM1, model_settings, train_mode_placeholder, '2')
    resCONVLSTM3 = resCONVLSTM(resCONVLSTM2, model_settings, train_mode_placeholder, '3')
    resCONVLSTM4= resCONVLSTM(resCONVLSTM3,model_settings,train_mode_placeholder,'4')
    resCONVLSTM4=tf.reshape(resCONVLSTM4,[-1, input_time_size, input_frequency_size])
    with tf.variable_scope('lstm1'):
        lstm_cell1=tf.contrib.rnn.LSTMCell(num_units=650,num_proj=input_frequency_size)
        initial_lstm1=lstm_cell1.zero_state(batch_size,dtype=tf.float32)
        lstm1_o,_=tf.nn.dynamic_rnn(lstm_cell1,resCONVLSTM4,initial_state=initial_lstm1)
        lstm1_o=tf.reshape(lstm1_o,[-1, input_time_size, input_frequency_size,1 ])
        nin1_o=tf.layers.conv2d(lstm1_o,1,[1,1],name='nin1')
        bn1=slim.layers.batch_norm(nin1_o,is_training=train_mode_placeholder,updates_collections=None,decay=0.96,
                                          zero_debias_moving_mean=True,activation_fn=tf.nn.relu,data_format='NCHW')
        bn1=tf.reshape(bn1,[-1,input_time_size,input_frequency_size])
    with tf.variable_scope('lstm2'):
        lstm_cell2=tf.contrib.rnn.LSTMCell(num_units=650,num_proj=input_frequency_size)
        initial_lstm2=lstm_cell1.zero_state(batch_size,dtype=tf.float32)
        lstm2_o, _ = tf.nn.dynamic_rnn(lstm_cell2, bn1, initial_state=initial_lstm2)
        lstm2_o = tf.reshape(lstm2_o, [-1, input_time_size, input_frequency_size, 1])
        nin2_o = tf.layers.conv2d(lstm2_o, 1, [1, 1], name='nin1')
        bn2 = slim.layers.batch_norm(nin2_o, is_training=train_mode_placeholder,updates_collections=None,decay=0.96,
                                          zero_debias_moving_mean=True,activation_fn=tf.nn.relu,data_format='NCHW')
        bn2=tf.reshape(bn2,[-1,input_time_size,input_frequency_size])

    # LSTM Layer Final
    with tf.variable_scope('lstm3'):
        lstm_cell3=tf.contrib.rnn.LSTMCell(num_units=650,num_proj=input_frequency_size)
        initial_lstm3=lstm_cell1.zero_state(batch_size,dtype=tf.float32)
        lstm3_o, _ = tf.nn.dynamic_rnn(lstm_cell3, bn2, initial_state=initial_lstm3)
        lstm3_o = tf.reshape(lstm3_o, [-1, input_time_size, input_frequency_size, 1])


    # Final FC for classification
    reshaped_layer = tf.reshape(lstm3_o,
                                 [-1, input_time_size * input_frequency_size])

    # Dropout
    if is_training:
        reshaped_layer = tf.nn.dropout(reshaped_layer, keep_prob=dropout_prob)


    prefinal_dense=tf.nn.relu(tf.layers.dense(reshaped_layer,750))

    if is_training:
        prefinal_dense=tf.nn.dropout(prefinal_dense,keep_prob=dropout_prob)
    # Final Layer

    label_count = model_settings['label_count']

    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [750, label_count], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(prefinal_dense, final_fc_weights) + final_fc_bias
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc



def create_lstm_l_model(fingerprint_input, model_settings, is_training):
    """
        Creates a LSTM Large Model with output projection


    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    batch_size = tf.shape(fingerprint_input)[0]
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    label_count = model_settings['label_count']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size])
    lstm_cell = tf.contrib.rnn.LSTMCell(500, use_peepholes=True,num_proj=188)
    if is_training:
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=dropout_prob)

    initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    _,states = tf.nn.dynamic_rnn(lstm_cell, fingerprint_4d, initial_state=initial_state)

    lstm_o = states[-1]

    # Final Output Layer
    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [188, label_count], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(lstm_o, final_fc_weights) + final_fc_bias
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def create_ds_cnn_model(fingerprint_input, model_settings, model_size_info,
                        is_training):
    """Builds a model with depthwise separable convolutional neural network
    Model definition is based on https://arxiv.org/abs/1704.04861 and
    Tensorflow implementation: https://github.com/Zehaos/MobileNet
    model_size_info: defines number of layers, followed by the DS-Conv layer
      parameters in the order {number of conv features, conv filter height, 
      width and stride in y,x dir.} for each of the layers. 
    Note that first layer is always regular convolution, but the remaining 
      layers are all depthwise separable convolutions.
    """

    def ds_cnn_arg_scope(weight_decay=0):
        """Defines the default ds_cnn argument scope.
        Args:
          weight_decay: The weight decay to use for regularizing the model.
        Returns:
          An `arg_scope` to use for the DS-CNN model.
        """
        with slim.arg_scope(
                [slim.convolution2d, slim.separable_convolution2d],
                weights_initializer=slim.initializers.xavier_initializer(),
                biases_initializer=slim.init_ops.zeros_initializer(),
                weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
            return sc

    def _depthwise_separable_conv(inputs,
                                  num_pwc_filters,
                                  sc,
                                  kernel_size,
                                  stride):
        """ Helper function to build the depth-wise separable convolution layer.
        """

        # skip pointwise by setting num_outputs=None
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                      num_outputs=None,
                                                      stride=stride,
                                                      depth_multiplier=1,
                                                      kernel_size=kernel_size,
                                                      scope=sc + '/depthwise_conv')

        bn = slim.batch_norm(depthwise_conv, scope=sc + '/dw_batch_norm')
        pointwise_conv = slim.convolution2d(bn,
                                            num_pwc_filters,
                                            kernel_size=[1, 1],
                                            scope=sc + '/pointwise_conv')
        bn = slim.batch_norm(pointwise_conv, scope=sc + '/pw_batch_norm')
        return bn

    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    label_count = model_settings['label_count']
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']

    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
  
    t_dim = input_time_size
    f_dim = input_frequency_size

    # Extract model dimensions from model_size_info
    num_layers = model_size_info[0]
    conv_feat = [None] * num_layers
    conv_kt = [None] * num_layers
    conv_kf = [None] * num_layers
    conv_st = [None] * num_layers
    conv_sf = [None] * num_layers
    i = 1
    for layer_no in range(0, num_layers):
        conv_feat[layer_no] = model_size_info[i]
        i += 1
        conv_kt[layer_no] = model_size_info[i]
        i += 1
        conv_kf[layer_no] = model_size_info[i]
        i += 1
        conv_st[layer_no] = model_size_info[i]
        i += 1
        conv_sf[layer_no] = model_size_info[i]
        i += 1

    scope = 'DS-CNN'
    with tf.variable_scope(scope) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                            activation_fn=None,
                            weights_initializer=slim.initializers.xavier_initializer(),
                            biases_initializer=slim.init_ops.zeros_initializer(),
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.batch_norm],
                                is_training=is_training,
                                decay=0.96,
                                updates_collections=None,
                                activation_fn=tf.nn.relu):
                for layer_no in range(0, num_layers):
                    if layer_no == 0:
                        net = slim.convolution2d(fingerprint_4d, conv_feat[layer_no], \
                                                 [conv_kt[layer_no], conv_kf[layer_no]],
                                                 stride=[conv_st[layer_no], conv_sf[layer_no]], padding='SAME',
                                                 scope='conv_1')
                        net = slim.batch_norm(net, scope='conv_1/batch_norm')
                    else:
                        net = _depthwise_separable_conv(net, conv_feat[layer_no], \
                                                        kernel_size=[conv_kt[layer_no], conv_kf[layer_no]], \
                                                        stride=[conv_st[layer_no], conv_sf[layer_no]],
                                                        sc='conv_ds_' + str(layer_no))
                    t_dim = math.ceil(t_dim / float(conv_st[layer_no]))
                    f_dim = math.ceil(f_dim / float(conv_sf[layer_no]))

                net = slim.avg_pool2d(net, [t_dim, f_dim], scope='avg_pool')

        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
        logits = slim.fully_connected(net, label_count, activation_fn=None, scope='fc1')

    if is_training:
        return logits, dropout_prob
    else:
        return logits


def create_crnn_model(fingerprint_input, model_settings,
                      model_size_info, is_training):
    """Builds a model with convolutional recurrent networks with GRUs
    Based on the model definition in https://arxiv.org/abs/1703.05390
    model_size_info: defines the following convolution layer parameters
        {number of conv features, conv filter height, width, stride in y,x dir.},
        followed by number of GRU layers and number of GRU cells per layer
    Optionally, the bi-directional GRUs and/or GRU with layer-normalization 
      can be explored.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])

    layer_norm = False
    bidirectional = False

    # CNN part
    first_filter_count = model_size_info[0]
    first_filter_height = model_size_info[1]
    first_filter_width = model_size_info[2]
    first_filter_stride_y = model_size_info[3]
    first_filter_stride_x = model_size_info[4]

    first_weights = tf.get_variable('W', shape=[first_filter_height,
                                                first_filter_width, 1, first_filter_count],
                                    initializer=tf.contrib.layers.xavier_initializer())

    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
        1, first_filter_stride_y, first_filter_stride_x, 1
    ], 'VALID') + first_bias
    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu
    first_conv_output_width = int(math.floor(
        (input_frequency_size - first_filter_width + first_filter_stride_x) /
        first_filter_stride_x))
    first_conv_output_height = int(math.floor(
        (input_time_size - first_filter_height + first_filter_stride_y) /
        first_filter_stride_y))

    # GRU part
    num_rnn_layers = model_size_info[5]
    RNN_units = model_size_info[6]
    flow = tf.reshape(first_dropout, [-1, first_conv_output_height,
                                      first_conv_output_width * first_filter_count])
    cell_fw = []
    cell_bw = []
    if layer_norm:
        for i in range(num_rnn_layers):
            cell_fw.append(LayerNormGRUCell(RNN_units))
            if bidirectional:
                cell_bw.append(LayerNormGRUCell(RNN_units))
    else:
        for i in range(num_rnn_layers):
            cell_fw.append(tf.contrib.rnn.GRUCell(RNN_units))
            if bidirectional:
                cell_bw.append(tf.contrib.rnn.GRUCell(RNN_units))

    if bidirectional:
        outputs, output_state_fw, output_state_bw = \
            tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cell_fw, cell_bw, flow,
                                                           dtype=tf.float32)
        flow_dim = first_conv_output_height * RNN_units * 2
        flow = tf.reshape(outputs, [-1, flow_dim])
    else:
        cells = tf.contrib.rnn.MultiRNNCell(cell_fw)
        _, last = tf.nn.dynamic_rnn(cell=cells, inputs=flow, dtype=tf.float32)
        flow_dim = RNN_units
        flow = last[-1]

    first_fc_output_channels = model_size_info[7]

    first_fc_weights = tf.get_variable('fcw', shape=[flow_dim,
                                                     first_fc_output_channels],
                                       initializer=tf.contrib.layers.xavier_initializer())

    first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
    first_fc = tf.nn.relu(tf.matmul(flow, first_fc_weights) + first_fc_bias)
    if is_training:
        final_fc_input = tf.nn.dropout(first_fc, dropout_prob)
    else:
        final_fc_input = first_fc

    label_count = model_settings['label_count']

    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [first_fc_output_channels, label_count], stddev=0.01))

    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def create_low_latency_conv_model(fingerprint_input, model_settings,
                                  is_training):
    """Builds a convolutional model with low compute requirements.
  
    This is roughly the network labeled as 'cnn-one-fstride4' in the
    'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
    http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
  
    Here's the layout of the graph:
  
    (fingerprint_input)
            v
        [Conv2D]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
          [Relu]
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
  
    This produces slightly lower quality results than the 'conv' model, but needs
    fewer weight parameters and computations.
  
    During training, dropout nodes are introduced after the relu, controlled by a
    placeholder.
  
    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.
  
    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
    first_filter_width = 8
    first_filter_height = input_time_size
    first_filter_count = 186
    first_filter_stride_x = 1
    first_filter_stride_y = 1
    first_weights = tf.Variable(
        tf.truncated_normal(
            [first_filter_height, first_filter_width, 1, first_filter_count],
            stddev=0.01))
    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
        1, first_filter_stride_y, first_filter_stride_x, 1
    ], 'VALID') + first_bias
    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu
    first_conv_output_width = math.floor(
        (input_frequency_size - first_filter_width + first_filter_stride_x) /
        first_filter_stride_x)
    first_conv_output_height = math.floor(
        (input_time_size - first_filter_height + first_filter_stride_y) /
        first_filter_stride_y)
    first_conv_element_count = int(
        first_conv_output_width * first_conv_output_height * first_filter_count)
    flattened_first_conv = tf.reshape(first_dropout,
                                      [-1, first_conv_element_count])
    first_fc_output_channels = 128
    first_fc_weights = tf.Variable(
        tf.truncated_normal(
            [first_conv_element_count, first_fc_output_channels], stddev=0.01))
    first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
    first_fc = tf.matmul(flattened_first_conv, first_fc_weights) + first_fc_bias
    if is_training:
        second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
    else:
        second_fc_input = first_fc
    second_fc_output_channels = 128
    second_fc_weights = tf.Variable(
        tf.truncated_normal(
            [first_fc_output_channels, second_fc_output_channels], stddev=0.01))
    second_fc_bias = tf.Variable(tf.zeros([second_fc_output_channels]))
    second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
    if is_training:
        final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
    else:
        final_fc_input = second_fc
    label_count = model_settings['label_count']
    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [second_fc_output_channels, label_count], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def create_low_latency_svdf_model(fingerprint_input, model_settings,
                                  is_training, runtime_settings):
    """Builds an SVDF model with low compute requirements.
  
    This is based in the topology presented in the 'Compressing Deep Neural
    Networks using a Rank-Constrained Topology' paper:
    https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43813.pdf
  
    Here's the layout of the graph:
  
    (fingerprint_input)
            v
          [SVDF]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
          [Relu]
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
  
    This model produces lower recognition accuracy than the 'conv' model above,
    but requires fewer weight parameters and, significantly fewer computations.
  
    During training, dropout nodes are introduced after the relu, controlled by a
    placeholder.
  
    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      The node is expected to produce a 2D Tensor of shape:
        [batch, model_settings['dct_coefficient_count'] *
                model_settings['spectrogram_length']]
      with the features corresponding to the same time slot arranged contiguously,
      and the oldest slot at index [:, 0], and newest at [:, -1].
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.
      runtime_settings: Dictionary of information about the runtime.
  
    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
  
    Raises:
        ValueError: If the inputs tensor is incorrectly shaped.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']

    # Validation.
    input_shape = fingerprint_input.get_shape()
    if len(input_shape) != 2:
        raise ValueError('Inputs to `SVDF` should have rank == 2.')
    if input_shape[-1].value is None:
        raise ValueError('The last dimension of the inputs to `SVDF` '
                         'should be defined. Found `None`.')
    if input_shape[-1].value % input_frequency_size != 0:
        raise ValueError('Inputs feature dimension %d must be a multiple of '
                         'frame size %d', fingerprint_input.shape[-1].value,
                         input_frequency_size)

    # Set number of units (i.e. nodes) and rank.
    rank = 2
    num_units = 1280
    # Number of filters: pairs of feature and time filters.
    num_filters = rank * num_units
    # Create the runtime memory: [num_filters, batch, input_time_size]
    batch = 1
    memory = tf.Variable(tf.zeros([num_filters, batch, input_time_size]),
                         trainable=False, name='runtime-memory')
    # Determine the number of new frames in the input, such that we only operate
    # on those. For training we do not use the memory, and thus use all frames
    # provided in the input.
    # new_fingerprint_input: [batch, num_new_frames*input_frequency_size]
    if is_training:
        num_new_frames = input_time_size
    else:
        window_stride_ms = int(model_settings['window_stride_samples'] * 1000 /
                               model_settings['sample_rate'])
        num_new_frames = tf.cond(
            tf.equal(tf.count_nonzero(memory), 0),
            lambda: input_time_size,
            lambda: int(runtime_settings['clip_stride_ms'] / window_stride_ms))
    new_fingerprint_input = fingerprint_input[
                            :, -num_new_frames * input_frequency_size:]
    # Expand to add input channels dimension.
    new_fingerprint_input = tf.expand_dims(new_fingerprint_input, 2)

    # Create the frequency filters.
    weights_frequency = tf.Variable(
        tf.truncated_normal([input_frequency_size, num_filters], stddev=0.01))
    # Expand to add input channels dimensions.
    # weights_frequency: [input_frequency_size, 1, num_filters]
    weights_frequency = tf.expand_dims(weights_frequency, 1)
    # Convolve the 1D feature filters sliding over the time dimension.
    # activations_time: [batch, num_new_frames, num_filters]
    activations_time = tf.nn.conv1d(
        new_fingerprint_input, weights_frequency, input_frequency_size, 'VALID')
    # Rearrange such that we can perform the batched matmul.
    # activations_time: [num_filters, batch, num_new_frames]
    activations_time = tf.transpose(activations_time, perm=[2, 0, 1])

    # Runtime memory optimization.
    if not is_training:
        # We need to drop the activations corresponding to the oldest frames, and
        # then add those corresponding to the new frames.
        new_memory = memory[:, :, num_new_frames:]
        new_memory = tf.concat([new_memory, activations_time], 2)
        tf.assign(memory, new_memory)
        activations_time = new_memory

    # Create the time filters.
    weights_time = tf.Variable(
        tf.truncated_normal([num_filters, input_time_size], stddev=0.01))
    # Apply the time filter on the outputs of the feature filters.
    # weights_time: [num_filters, input_time_size, 1]
    # outputs: [num_filters, batch, 1]
    weights_time = tf.expand_dims(weights_time, 2)
    outputs = tf.matmul(activations_time, weights_time)
    # Split num_units and rank into separate dimensions (the remaining
    # dimension is the input_shape[0] -i.e. batch size). This also squeezes
    # the last dimension, since it's not used.
    # [num_filters, batch, 1] => [num_units, rank, batch]
    outputs = tf.reshape(outputs, [num_units, rank, -1])
    # Sum the rank outputs per unit => [num_units, batch].
    units_output = tf.reduce_sum(outputs, axis=1)
    # Transpose to shape [batch, num_units]
    units_output = tf.transpose(units_output)

    # Appy bias.
    bias = tf.Variable(tf.zeros([num_units]))
    first_bias = tf.nn.bias_add(units_output, bias)

    # Relu.
    first_relu = tf.nn.relu(first_bias)

    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu

    first_fc_output_channels = 256
    first_fc_weights = tf.Variable(
        tf.truncated_normal([num_units, first_fc_output_channels], stddev=0.01))
    first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
    first_fc = tf.matmul(first_dropout, first_fc_weights) + first_fc_bias
    if is_training:
        second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
    else:
        second_fc_input = first_fc
    second_fc_output_channels = 256
    second_fc_weights = tf.Variable(
        tf.truncated_normal(
            [first_fc_output_channels, second_fc_output_channels], stddev=0.01))
    second_fc_bias = tf.Variable(tf.zeros([second_fc_output_channels]))
    second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
    if is_training:
        final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
    else:
        final_fc_input = second_fc
    label_count = model_settings['label_count']
    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [second_fc_output_channels, label_count], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


class LayerNormGRUCell(rnn_cell_impl.RNNCell):
    def __init__(self, num_units, forget_bias=1.0,
                 input_size=None, activation=math_ops.tanh,
                 layer_norm=True, norm_gain=1.0, norm_shift=0.0,
                 dropout_keep_prob=1.0, dropout_prob_seed=None,
                 reuse=None):

        super(LayerNormGRUCell, self).__init__(_reuse=reuse)

        if input_size is not None:
            tf.logging.info("%s: The input_size parameter is deprecated.", self)

        self._num_units = num_units
        self._activation = activation
        self._forget_bias = forget_bias
        self._keep_prob = dropout_keep_prob
        self._seed = dropout_prob_seed
        self._layer_norm = layer_norm
        self._g = norm_gain
        self._b = norm_shift
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def _norm(self, inp, scope):
        shape = inp.get_shape()[-1:]
        gamma_init = init_ops.constant_initializer(self._g)
        beta_init = init_ops.constant_initializer(self._b)
        with vs.variable_scope(scope):
            # Initialize beta and gamma for use by layer_norm.
            vs.get_variable("gamma", shape=shape, initializer=gamma_init)
            vs.get_variable("beta", shape=shape, initializer=beta_init)
        normalized = layers.layer_norm(inp, reuse=True, scope=scope)
        return normalized

    def _linear(self, args, copy):
        out_size = copy * self._num_units
        proj_size = args.get_shape()[-1]
        weights = vs.get_variable("kernel", [proj_size, out_size])
        out = math_ops.matmul(args, weights)
        if not self._layer_norm:
            bias = vs.get_variable("bias", [out_size])
            out = nn_ops.bias_add(out, bias)
        return out

    def call(self, inputs, state):
        """LSTM cell with layer normalization and recurrent dropout."""
        with vs.variable_scope("gates"):
            h = state
            args = array_ops.concat([inputs, h], 1)
            concat = self._linear(args, 2)

            z, r = array_ops.split(value=concat, num_or_size_splits=2, axis=1)
            if self._layer_norm:
                z = self._norm(z, "update")
                r = self._norm(r, "reset")

        with vs.variable_scope("candidate"):
            args = array_ops.concat([inputs, math_ops.sigmoid(r) * h], 1)
            new_c = self._linear(args, 1)
            if self._layer_norm:
                new_c = self._norm(new_c, "state")
        new_h = self._activation(new_c) * math_ops.sigmoid(z) + \
                (1 - math_ops.sigmoid(z)) * h
        return new_h, new_h

