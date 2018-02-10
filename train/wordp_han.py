import os
import sys
sys.path.append("..")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

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

train = pd.read_csv(Config.cache_dir+"/train.csv",sep="\t")
val = pd.read_csv(Config.cache_dir+"/test.csv",sep="\t")
val_label = to_categorical(val.label)

val_word_seq = pickle.load(open(Config.cache_dir+"/val_word_han_seq_%s.pkl"%(Config.sentence_num*Config.sentence_word_length), "rb"))
val_wordp_seq = pickle.load(open(Config.cache_dir+"/val_wordp_han_seq_%s.pkl"%(Config.sentence_num*Config.sentence_word_length), "rb"))

batch_size = 128
model_name = "wordp_han"
trainable_layer = ["word_embedding"]
trainable_layer = []
train_batch_generator = wordp_han_train_batch_generator
val_seq = [val_word_seq, val_wordp_seq]

word_embed_weight = np.load(Config.word_embed_weight_path)
model = get_wordp_han(Config.sentence_num, Config.sentence_word_length, word_embed_weight)

for i in range(15):
    if i==6:
        K.set_value(model.optimizer.lr, 0.0001)
    if i==9:
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
