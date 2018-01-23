import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append("..")
from config import *
import codecs
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from joblib import Parallel,delayed
from utils.data import *

overwrite = True
vec_dim = 256
if overwrite:
    if os.path.exists(Config.cache_dir+"/w2v_dataframe.csv"):
        os.remove(Config.cache_dir+"/w2v_dataframe.csv")

    if os.path.exists(Config.cache_dir+"/w2v_content_word.txt"):
        os.remove(Config.cache_dir+"/w2v_content_word.txt")

    if os.path.exists(Config.cache_dir+"/w2v_content_char.txt"):
        os.remove(Config.cache_dir+"/w2v_content_char.txt")

    train_final = get_train_final_data()
    test_final = get_test_final_data()
    test_final["label"] = 0

    w2v_dataframe = pd.concat([train_final, test_final])[["id","content"]]
    w2v_dataframe.to_csv(Config.cache_dir+"/w2v_dataframe.csv",index=False,sep="\t")

    def applyParallel(contents, func, n_thread):
        with Parallel(n_jobs=n_thread) as parallel:
            parallel(delayed(func)(c) for c in contents)

    def word_cut(content):
        with open(Config.cache_dir+"/w2v_content_word.txt","a+") as f:
            f.writelines(" ".join(jieba.cut(content)))
            f.writelines("\n")

    def char_cut(content):
        with open(Config.cache_dir+"/w2v_content_char.txt","a+") as f:
            f.writelines(" ".join(content))
            f.writelines("\n")

    contents = w2v_dataframe.content.values
    applyParallel(contents, char_cut, 25)
    applyParallel(contents, word_cut, 25)

## word vector Train
model = Word2Vec(
    LineSentence(Config.cache_dir+"/w2v_content_word.txt"), size=vec_dim, window=5, min_count=5, workers=multiprocessing.cpu_count()
)
model.save(Config.cache_dir+"/360_content_w2v_word.model")

weights = model.wv.syn0
vocab = dict([(k, v.index+1) for k,v in model.wv.vocab.items()])

embed_weights = np.zeros(shape=(weights.shape[0]+2, weights.shape[1]))
embed_weights[1:weights.shape[0]+1] = weights
unk_vec = np.random.random(size=weights.shape[1])*0.5
embed_weights[weights.shape[0]+1] = unk_vec - unk_vec.mean()

pickle.dump(vocab, open(Config.cache_dir + "/word_embed_.dict.pkl", "wb"))
np.save(Config.cache_dir+"/word_embed_.npy", embed_weights)

## char vector Train
model = Word2Vec(
    LineSentence(Config.cache_dir+"/w2v_content_char.txt"), size=vec_dim, window=5, min_count=5, workers=multiprocessing.cpu_count()
)
model.save(Config.cache_dir+"/360_content_w2v_char.model")

weights = model.wv.syn0
vocab = dict([(k, v.index+1) for k,v in model.wv.vocab.items()])

embed_weights = np.zeros(shape=(weights.shape[0]+2, weights.shape[1]))
embed_weights[1:weights.shape[0]+1] = weights
unk_vec = np.random.random(size=weights.shape[1])*0.5
embed_weights[weights.shape[0]+1] = unk_vec - unk_vec.mean()

pickle.dump(vocab, open(Config.cache_dir + "/char_embed_.dict.pkl", "wb"))
np.save(Config.cache_dir+"/char_embed_.npy", embed_weights)
