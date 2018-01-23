import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from config import *
import keras
import keras.backend as K
from utils.data import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.utils import to_categorical
from utils.others import *

pred_list = [
    "%s_char_cnn.pred",
    "%s_word_cnn.pred",
    "%s_word_char_cnn.pred",
]
val = pd.read_csv(Config.cache_dir+"/val.csv", sep="\t")
val_label = to_categorical(val.label)

model_count = len(pred_list)
data = np.zeros(shape=(val.shape[0], 2, model_count))
for i,v in enumerate(pred_list):
    data[:,:,i] = np.load(Config.cache_dir + "/" + v%"val")

datainput = Input(shape=(2, model_count))
output = Activation(activation="softmax")(Reshape((2,))(Conv1D(1, 1, kernel_initializer="ones", use_bias=True, name="model_weight")(datainput)))
model = Model(inputs=datainput,outputs=output)
model.compile(loss='categorical_crossentropy',optimizer="sgd",metrics=['accuracy'])

model.fit(data, val_label, validation_split=0.1, epochs=12)

pred = model.predict(data)
p,r,f1 = score(pred, val_label)
print(p,r,f1)
print(model.get_layer("model_weight").get_weights())

test_final = get_test_final_data()
test_data = np.zeros(shape=(test_final.shape[0], 2, model_count))
for i,v in enumerate(pred_list):
    test_data[:,:,i] = np.load(Config.common_dir + "/" + v%"test_final")

test_pred = model.predict(test_data)
test_final["pred"] = np.argmax(test_pred, axis=1)
submit(test_final)
print(test_final[test_final.pred == "POSITIVE"].shape)
