import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import sys
import keras
import keras.backend as K
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.utils import to_categorical
from utils.data import *
from utils.others import *

## Val Data Load
print("val")
val = pd.read_csv(Config.cache_dir+"/val.csv", sep="\t")
val_label = to_categorical(val.label)
val_word_seq = pickle.load(open(Config.cache_dir+"/g_val_word_seq_%s.pkl"%Config.word_seq_maxlen, "rb"))
val_word_han_seq = pickle.load(open(Config.cache_dir+"/g_val_word_han_seq_%s.pkl"%(Config.sentence_num*Config.sentence_word_length), "rb"))
val_char_seq = pickle.load(open(Config.cache_dir+"/g_val_char_seq_%s.pkl"%Config.char_seq_maxlen, "rb"))
val_wordp_seq = pickle.load(open(Config.cache_dir+"/g_val_wordp_seq_%s.pkl"%Config.word_seq_maxlen, "rb"))

print("test final")
## Test Data Load
test_final_word_seq = pickle.load(open(Config.cache_dir+"/g_test_final_word_seq_%s.pkl"%Config.word_seq_maxlen, "rb"))
test_final_word_han_seq_1 = pickle.load(open(Config.cache_dir+"/g_test_final_word_han_seq_%s_1.pkl"%(Config.sentence_num*Config.sentence_word_length), "rb"))
test_final_word_han_seq_2 = pickle.load(open(Config.cache_dir+"/g_test_final_word_han_seq_%s_2.pkl"%(Config.sentence_num*Config.sentence_word_length), "rb"))
test_final_word_han_seq = np.concatenate([test_final_word_han_seq_1, test_final_word_han_seq_2])
test_final_char_seq = pickle.load(open(Config.cache_dir+"/g_test_final_char_seq_%s.pkl"%Config.char_seq_maxlen, "rb"))
test_final_wordp_seq = pickle.load(open(Config.cache_dir+"/g_test_final_wordp_seq_%s.pkl"%Config.word_seq_maxlen, "rb"))

## Model Load
model_name = "model_name"
model_weight = "/model/weight/path"
model_path = Config.weight_dir + "/" + model_weight
print(model_name, model_path)

#### wordp_char
val_seq = [val_word_seq, val_wordp_seq, val_char_seq]
test_seq = [test_final_word_seq, test_final_wordp_seq, test_final_char_seq]

#### word_char
val_seq = [val_word_seq, val_char_seq]
test_seq = [test_final_word_seq, test_final_char_seq]

### word
val_seq = val_word_han_seq
test_seq = test_final_word_han_seq

### word
val_seq = val_word_seq
test_seq = test_final_word_seq

### char
val_seq = val_char_seq
test_seq = test_final_char_seq

model = load_model(model_path)

## Val Data Pred
val_model_pred = np.squeeze(model.predict(val_seq))
pickle.dump(val_model_pred,open(Config.cache_dir+"/val_%s.pred"%model_name,"wb"))
val_model_pred = pickle.load(open(Config.cache_dir+"/val_%s.pred"%model_name,"rb"))

p,r,f = score(val_model_pred, val_label)
print(p,r,f)

## Test Data Pred
test_model_pred = np.squeeze(model.predict(test_seq))
pickle.dump(test_model_pred, open(Config.cache_dir+"/test_final_%s.pred"%model_name,"wb"))
