from keras.layers.convolutional import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape,Masking,Lambda,Permute
from keras.layers import Input,Dense,Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.layers.recurrent import GRU,LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam,SGD,Adadelta
from keras import losses
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
from keras.utils import plot_model
from matplotlib import pyplot as plt

import numpy as np 
import os
from PIL import Image,ImageDraw,ImageFont 
import json
import threading
import pandas as pd
from opencc import OpenCC 

import tensorflow as tf  
import keras.backend.tensorflow_backend as K  


name_corpus = pd.read_csv("./input/chinese_names_big5.csv")

name_corpus.columns=['name_sim_chi','len','name_tra_chi']
name_corpus.drop_duplicates(inplace=True)

name_set=list(set(name_corpus['name_tra_chi']))

char_list=[]

for name in name_set:
    char_list.extend(list(name))
    
char_to_id = {j:i for i,j in enumerate(char_list)}
id_to_char = {i:j for i,j in enumerate(char_list)}

char_df = pd.DataFrame(data=char_list)
char_df.columns=['char']
char_df['count']=1
char_stat = char_df.groupby('char').sum().sort_values(by='count',ascending=False)


maxlabellength = 3
img_h = 32
img_w = 248
nclass = len(char_stat)
rnnunit=256
batch_size =64


font=ImageFont.truetype('/usr/share/fonts/truetype/windows/mingliu0.ttf',24) 

def generate_image_sample(n=100,del_file_path='./train/*',image_path='./train/train_',label_path = "./train/train_label.csv"):
    
    import os
    #os.remove(del_file_path)
    sample_name=name_corpus.sample(n)
    for index, row in sample_name.iterrows():

        img = img = Image.new('L',(img_w,img_h),(255))
        draw = ImageDraw.Draw(img)  
        name = row['name_tra_chi']
        label = ""
        for chr in name:
            label = label + chr +" "
        draw.text((0,5),label.strip() ,fill=(0),font=font)  
        img.save(image_path+str(index)+'.png')

    sample_name.reset_index().to_csv(label_path,index=False)  
    return sample_name

train_image = generate_image_sample(len(name_corpus),'./train/*.*','./train/train_','train/train_label.csv')
y_train = train_image['name_tra_chi']

