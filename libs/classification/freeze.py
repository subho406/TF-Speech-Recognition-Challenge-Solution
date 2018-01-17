# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Converts a trained checkpoint into a frozen model for mobile inference.
Once you've trained a model using the `train.py` script, you can use this tool
to convert it into a binary GraphDef file that can be loaded into the Android,
iOS, or Raspberry Pi example code. Here's an example of how to run it:
bazel run tensorflow/examples/speech_commands/freeze -- \
--sample_rate=16000 --dct_coefficient_count=40 --window_size_ms=20 \
--window_stride_ms=10 --clip_duration_ms=1000 \
--model_architecture=conv \
--start_checkpoint=/tmp/speech_commands_train/conv.ckpt-1300 \
--output_file=/tmp/my_frozen_graph.pb
One thing to watch out for is that you need to pass in the same arguments for
`sample_rate` and other command line variables here as you did for the training
script.
The resulting graph has an input for WAV-encoded data named 'wav_data', one for
raw PCM data (as floats in the range -1.0 to 1.0) called 'decoded_sample_data',
and the output is called 'labels_softmax'.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf
from tensorflow.contrib import ffmpeg as ffmpeg
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from . import input_data
from . import models
from tensorflow.python.framework import graph_util


def create_inference_graph(wanted_words, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count, model_architecture,model_size_info=None):
    """Creates an audio model with the nodes needed for inference.
    Uses the supplied arguments to create a model, and inserts the input and
    output nodes that are needed to use the graph for inference.
    Args:
      wanted_words: Comma-separated list of the words we're trying to recognize.
      sample_rate: How many samples per second are in the input audio files.
      clip_duration_ms: How many samples to analyze for the audio pattern.
      clip_stride_ms: How often to run recognition. Useful for models with cache.
      window_size_ms: Time slice duration to estimate frequencies from.
      window_stride_ms: How far apart time slices should be.
      dct_coefficient_count: Number of frequency bands to analyze.
      model_architecture: Name of the kind of model to generate.
    """

    words_list = input_data.prepare_words_list(wanted_words.split(','))
    model_settings = models.prepare_model_settings(
        len(words_list), sample_rate, clip_duration_ms, window_size_ms,
        window_stride_ms, dct_coefficient_count)
    if(model_architecture=='dnc'):
        model_settings['batch_size']=1
                   
    wav_data_placeholder = tf.placeholder(tf.string, [], name='wav_data')
    audio_binary=tf.read_file(wav_data_placeholder)
    decoded_sample_data = ffmpeg.decode_audio(audio_binary, file_format='wav', samples_per_second=model_settings['desired_samples'], channel_count=1)
    decoded_sample_data=tf.reshape(decoded_sample_data,shape=(model_settings['desired_samples']))
    spectrogram = contrib_audio.audio_spectrogram(
        decoded_sample_data,
        window_size=model_settings['window_size_samples'],
        stride=model_settings['window_stride_samples'],
        magnitude_squared=True)
    fingerprint_input = contrib_audio.mfcc(
        spectrogram,
        decoded_sample_data.sample_rate,
        dct_coefficient_count=dct_coefficient_count)
    fingerprint_frequency_size = model_settings['dct_coefficient_count']
    fingerprint_time_size = model_settings['spectrogram_length']
    reshaped_input = tf.reshape(fingerprint_input, [
        -1, fingerprint_time_size * fingerprint_frequency_size
    ])

    logits = models.create_model(
        reshaped_input, model_settings, model_architecture, model_size_info=model_size_info, is_training=False)

    # Create an output to use for inference.
    tf.nn.softmax(logits, name='labels_softmax')

    
def create_inference_graph_batched(wanted_words, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count, model_architecture,model_size_info=None):
    """Creates an audio model with the nodes needed for inference.
    Uses the supplied arguments to create a model, and inserts the input and
    output nodes that are needed to use the graph for inference.
    Args:
      wanted_words: Comma-separated list of the words we're trying to recognize.
      sample_rate: How many samples per second are in the input audio files.
      clip_duration_ms: How many samples to analyze for the audio pattern.
      clip_stride_ms: How often to run recognition. Useful for models with cache.
      window_size_ms: Time slice duration to estimate frequencies from.
      window_stride_ms: How far apart time slices should be.
      dct_coefficient_count: Number of frequency bands to analyze.
      model_architecture: Name of the kind of model to generate.
    """

    words_list = input_data.prepare_words_list(wanted_words.split(','))
    model_settings = models.prepare_model_settings(
        len(words_list), sample_rate, clip_duration_ms, window_size_ms,
        window_stride_ms, dct_coefficient_count)
    if(model_architecture=='dnc'):
        model_settings['batch_size']=1000
    fingerprint_size = model_settings['fingerprint_size']
    #Wav Data Placeholder
    wav_data_placeholder = tf.placeholder(tf.string, [], name='wav_data')
    decoded_sample_data = contrib_audio.decode_wav(
        wav_data_placeholder,
        desired_channels=1,
        desired_samples=model_settings['desired_samples'],
        name='decoded_sample_data')
    spectrogram = contrib_audio.audio_spectrogram(
        decoded_sample_data.audio,
        window_size=model_settings['window_size_samples'],
        stride=model_settings['window_stride_samples'],
        magnitude_squared=True)
    mfcc_output = contrib_audio.mfcc(
        spectrogram,
        decoded_sample_data.sample_rate,
        dct_coefficient_count=dct_coefficient_count,name='mfcc')
    #Batched Input Placeholder
    fingerprint_input = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')
    
    fingerprint_frequency_size = model_settings['dct_coefficient_count']
    fingerprint_time_size = model_settings['spectrogram_length']
    
    reshaped_input = tf.reshape(fingerprint_input, [
        -1, fingerprint_time_size * fingerprint_frequency_size
    ])

    logits = models.create_model(
        reshaped_input, model_settings, model_architecture, model_size_info=model_size_info, is_training=False)

    # Create an output to use for inference.
    tf.nn.softmax(logits, name='labels_softmax')



    
def freeze_graph(FLAGS, model_architecture, checkpoint_file, output_file,model_size_info=None,batched=False):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    if batched==False:
        create_inference_graph(FLAGS.wanted_words, FLAGS.sample_rate,
                               FLAGS.clip_duration_ms,
                               FLAGS.window_size_ms, FLAGS.window_stride_ms,
                               FLAGS.dct_coefficient_count, model_architecture,model_size_info)
    else:
        create_inference_graph_batched(FLAGS.wanted_words, FLAGS.sample_rate,
                               FLAGS.clip_duration_ms,
                               FLAGS.window_size_ms, FLAGS.window_stride_ms,
                               FLAGS.dct_coefficient_count, model_architecture,model_size_info)
        
    models.load_variables_from_checkpoint(sess, checkpoint_file)

    # Turn all the variables into inline constants inside the graph and save it.
    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ['labels_softmax','mfcc'])
    tf.train.write_graph(
        frozen_graph_def,
        os.path.dirname(output_file),
        os.path.basename(output_file),
        as_text=False)
    tf.logging.info('Saved frozen graph to %s', output_file)
    sess.close()
