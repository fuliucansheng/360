import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import sys
sys.path.append("..")
from utils.preprocess import *
import keras
import keras.backend as K
from utils.data import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from utils.others import *
from models.deepzoo import *

print("Load Train && Val")
train = pd.read_csv(Config.cache_dir+"/train.csv",sep="\t")
val = pd.read_csv(Config.cache_dir+"/val.csv",sep="\t")
val_label = to_categorical(val.label)

batch_size = 64
model_name = "char"
trainable_layer = ["embedding"]
train_batch_generator = char_cnn_train_batch_generator

print("Load Val Data")
val_char_seq = pickle.load(open(Config.cache_dir+"/g_val_char_seq_%s.pkl"%Config.char_seq_maxlen, "rb"))
val_seq = val_char_seq

print("Load Word")
char_embed_weight = np.load(Config.char_embed_weight_path)

model = get_textcnn(Config.char_seq_maxlen, char_embed_weight)

for i in range(15):
    if i==8:
        K.set_value(model.optimizer.lr, 0.0001)
    if i==12:
        for l in trainable_layer:
            model.get_layer(l).trainable = True
    model.fit_generator(
        train_batch_generator(train.content.values, train.label.values, batch_size=batch_size),
        epochs = 1,
        steps_per_epoch = int(train.shape[0]/batch_size),
        validation_data = (val_seq, val_label)
    )
    pred = np.squeeze(model.predict(val_seq))
    pre,rec,f1 = score(pred, val_label)
    print(pre,rec,f1)
    model.save(Config.cache_dir + "/dp_embed_%s_epoch_%s_%s.h5"%(model_name, i, f1))
