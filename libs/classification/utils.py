# Cognitbit Solutions LLP
#
# Author: Subhojeet Pramanik
# ==============================================================================
"""
This script contains a basic template for:
    1. Spliting the dataset into train, test and validation sets
    2. Generating a sample dataset
    3. Printing the sample dataset
    4. Cleaning
"""
import os
import numpy as np
import shutil

from shutil import copy2
from tensorflow.python.platform import gfile


def copy_dir(src, dst, *, follow_sym=True):
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    if os.path.isdir(src):
        shutil.copyfile(src, dst, follow_symlinks=follow_sym)
        shutil.copystat(src, dst, follow_symlinks=follow_sym)
    return dst


class Dataset:
    
    def __init__(self,data_dir):
        """
        Initialize the Dataset
        """
        self.data_dir=data_dir
        return None

    def split_train_test_val(self):
        """
        Split Dataset into Train Test and Validation Set
        """
        print('Splitting not required as we are going to use a hash based approach to get train, \
                test and val sets on the fly')

    def gen_sample_set(self,sample_dir,sample_ratio=0.01):
        """
        Generate the sample Dataset
        """
        print('Make sure to extract train and test achives as train and test folders in data_dir')
        if(os.path.exists(sample_dir)):
            shutil.rmtree(sample_dir)
        shutil.copytree(self.data_dir, sample_dir, copy_function=copy_dir)
        
       
        #Subset of Train
        #Copy Background noise as it is
        shutil.rmtree(os.path.join(sample_dir,'train','audio','_background_noise_'))
        shutil.copytree(os.path.join(self.data_dir,'train','audio','_background_noise_'),
                       os.path.join(sample_dir,'train','audio','_background_noise_'),ignore=None)
        search_path = os.path.join(self.data_dir, 'train','audio','*', '*.wav')
        wav_list=[]
        for wav_path in gfile.Glob(search_path):
            wav_list.append(wav_path)
        wav_list=np.array(wav_list)
        sample_size=int(sample_ratio*len(wav_list))
        chosen_wavs=np.random.choice(wav_list,sample_size)
        for c in chosen_wavs:
            splt=c.split('/')
            folder_name=splt[-2]
            file_name=splt[-1]
            save_path=os.path.join(sample_dir,'train','audio',folder_name,file_name)
            copy2(c,save_path)
        #Subset of Test
        search_path = os.path.join(self.data_dir, 'test','audio', '*.wav')
        wav_list=[]
        for wav_path in gfile.Glob(search_path):
            wav_list.append(wav_path)
        wav_list=np.array(wav_list)
        sample_size=int(sample_ratio*len(wav_list))
        chosen_wavs=np.random.choice(wav_list,sample_size)
        for c in chosen_wavs:
            splt=c.split('/')
            file_name=splt[-1]
            save_path=os.path.join(sample_dir,'test','audio',file_name)
            copy2(c,save_path)
        print('Subset Creation Successful!')

    def print_dataset_stats(self):
        '''
        Print the Dataset Statistics
        '''
        #Print number of files in Training and Test Set
        search_path = os.path.join(self.data_dir, 'train','audio','*', '*.wav')
        wav_list=[]
        count_train_wavs=len(gfile.Glob(search_path))
        print('Training WAV count: %d'%count_train_wavs)
        search_path = os.path.join(self.data_dir, 'test','audio','*.wav')
        wav_list=[]
        count_test_wavs=len(gfile.Glob(search_path))
        print('Test WAV count: %d'%count_test_wavs)

    def clean(self):
        '''
        Clean the Dataset
        '''
        return True