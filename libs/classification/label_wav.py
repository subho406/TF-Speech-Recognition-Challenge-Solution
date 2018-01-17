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
"""Runs a trained audio graph against a WAVE file and reports the results.

The model, labels and .wav file specified in the arguments will be loaded, and
then the predictions from running the model against the audio data will be
printed to the console. This is a useful script for sanity checking trained
models, and as an example of how to use an audsysio model from Python.

Here's an example of running it:

python tensorflow/examples/speech_commands/label_wav.py \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import ffmpeg
import argparse
import glob
import os
import numpy as np
import tensorflow as tf
import pandas as pd

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

FLAGS = None


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def run_graph(files_list, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  """Runs the audio data through the graph and prints predictions."""
  with tf.Session() as sess:
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    f_list=[]
    pred_list=[]
    total_files=len(files_list)
    count=0
    for wav in files_list:
        with open(wav, 'rb') as wav_file:
             wav_data = wav_file.read()
        predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

        # Sort to show labels in order of confidence
        pred_max = predictions.argmax()
        human_string = labels[pred_max]
        file_name=os.path.basename(wav)
        f_list.append(file_name)
        pred_list.append(human_string)
        count=count+1
        print('Completed %0.2f%% '%((count/total_files)*100),end='\r')
    return f_list,pred_list


def label_wav(files_list,labels_list, graph, input_name, output_name, how_many_labels):
  """Loads the model and labels, and runs the inference to print predictions."""

  # load graph, which is stored in the default session
  load_graph(graph)
    
  print('Number of Files %d'%(len(files_list)))


  f,p=run_graph(files_list, labels_list, input_name, output_name, how_many_labels)
  final_df=pd.DataFrame({'fname':f,'label':p})
  return final_df

def label_wav_batched(files_list,labels_list,graph,batch_size=10):
    """
    Label the wav in batched fashion
    """
    
    load_graph(graph)
    
    print('Number of Files %d'%(len(files_list)))
    
    with tf.Session() as sess:
        mfcc_tensor=sess.graph.get_tensor_by_name('mfcc:0')
        softmax_tensor = sess.graph.get_tensor_by_name('labels_softmax:0')
        all_predictions=[]
        prediction_labels=[]
        for i in range(0,len(files_list),batch_size):
            #Get the mfcc values of this batch
            batch_len=min(len(files_list)-i,batch_size)
            
            mfcc_values=[]
            
            for j in range(0,batch_len):
                wav=files_list[i+j]
                with open(wav, 'rb') as wav_file:
                    wav_data = wav_file.read()
                mfcc, = sess.run(mfcc_tensor,{'wav_data:0':wav_data})
                mfcc_values.append(mfcc)
            mfcc_values=np.stack(mfcc_values,axis=0)
            
            mfcc_values=np.reshape(mfcc_values,(-1,mfcc_values.shape[1]*mfcc_values.shape[2]))
            #We need this fix because many models support only fixed batch sizes
            if batch_len<batch_size:
                mfcc_values=np.concatenate([mfcc_values,np.zeros([batch_size-batch_len,mfcc_values.shape[1]])],axis=0)
            predictions = sess.run(softmax_tensor, {'fingerprint_input:0': mfcc_values})
            
            all_predictions.append(predictions)
            
            pred_max = predictions.argmax(axis=1)
            
            for j in range(0,batch_len):
                prediction_labels.append(labels_list[pred_max[j]])
            process_perct=((i+batch_len)/len(files_list))*100
            print('Processed %f%%'%process_perct)
        all_predictions=np.concatenate(all_predictions,axis=0)
        all_predictions=all_predictions[0:len(files_list)]
        return all_predictions,np.array(prediction_labels)


            
def label_mfcc_batched(mfcc,labels_list,graph,batch_size=10):
    load_graph(graph)
    
    print('Number of Files %d'%(mfcc.shape[0]))
    
    with tf.Session() as sess:
        mfcc_tensor=sess.graph.get_tensor_by_name('mfcc:0')
        softmax_tensor = sess.graph.get_tensor_by_name('labels_softmax:0')
        all_predictions=[]
        prediction_labels=[]
        for i in range(0,mfcc.shape[0],batch_size):
            #Get the mfcc values of this batch
            batch_len=min(mfcc.shape[0]-i,batch_size)
            
            mfcc_values=mfcc[i:i+batch_len]
            
            #We need this fix because many models support only fixed batch sizes
            if batch_len<batch_size:
                mfcc_values=np.concatenate([mfcc_values,np.zeros([batch_size-batch_len,mfcc_values.shape[1]])],axis=0)
            predictions = sess.run(softmax_tensor, {'fingerprint_input:0': mfcc_values})
            
            all_predictions.append(predictions)
            
            pred_max = predictions.argmax(axis=1)
            
            for j in range(0,batch_len):
                prediction_labels.append(labels_list[pred_max[j]])
            process_perct=((i+batch_len)/mfcc.shape[0])*100
            print('Processed %f%%'%process_perct)
        all_predictions=np.concatenate(all_predictions,axis=0)
        all_predictions=all_predictions[0:mfcc.shape[0]]
        return all_predictions,np.array(prediction_labels)
                       