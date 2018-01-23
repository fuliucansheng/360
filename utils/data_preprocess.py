import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append("..")
from config import *
from utils.data import *
from utils.preprocess import *
from sklearn.model_selection import train_test_split

def gen_word_property_dict():
    word_table = pd.read_csv(Config.data_dir+"/libs/dict.txt.big", sep=" ", header=None, names=["word", "fre", "property"])
    word_property = word_table["property"].drop_duplicates().values
    word_property_dict = dict(zip(word_property, np.arange(len(word_property))+1))
    pickle.dump(word_property_dict, open(Config.cache_dir + "/wordp_embed.dict.pkl", "wb"))

gen_word_property_dict()

train_data = get_train_final_data()
train, val = train_test_split(train_data, test_size=0.1)
train.to_csv(Config.cache_dir+"/train.csv", index=False)
val.to_csv(Config.cache_dir+"/val.csv", index=False)

## val
val = pd.read_csv(Config.cache_dir+"/val.csv", sep="\t")
val_char_seq = get_char_seq(val.content.values)
gc.collect()
pickle.dump(val_char_seq, open(Config.cache_dir+"/g_val_char_seq_%s.pkl"%Config.char_seq_maxlen, "wb"))

val_word_han_seq, val_wordp_han_seq = wordp_han_preprocess(val.content.values)
gc.collect()
pickle.dump(val_word_han_seq, open(Config.cache_dir+"/g_val_word_han_seq_%s.pkl"%(Config.sentence_num*Config.sentence_word_length), "wb"))
pickle.dump(val_wordp_han_seq, open(Config.cache_dir+"/g_val_wordp_han_seq_%s.pkl"%(Config.sentence_num*Config.sentence_word_length),"wb"))

val_word_seq,val_wordp_seq = get_wordp_seq(val.content.values)
gc.collect()
pickle.dump(val_word_seq, open(Config.cache_dir+"/g_val_word_seq_%s.pkl"%Config.word_seq_maxlen, "wb"))
pickle.dump(val_wordp_seq, open(Config.cache_dir+"/g_val_wordp_seq_%s.pkl"%Config.word_seq_maxlen, "wb"))

val_word_seq_plus,val_wordp_seq_plus = get_wordp_seq(val.content.values, word_maxlen=word_seq_maxlen_plus)
gc.collect()
pickle.dump(val_word_seq_plus, open(Config.cache_dir+"/g_val_word_seq_%s.pkl"%Config.word_seq_maxlen_plus,"wb"))
pickle.dump(val_wordp_seq_plus, open(Config.cache_dir+"/g_val_wordp_seq_%s.pkl"%Config.word_seq_maxlen_plus,"wb"))

### test_final
test_final = get_test_final_data()
test_final_char_seq = get_char_seq(test_final.content.values)
gc.collect()
pickle.dump(test_final_char_seq, open(Config.cache_dir+"/g_test_final_char_seq_%s.pkl"%Config.char_seq_maxlen,"wb"))

### 测试集比较大，分batch生成
batch_size = 100000
test_words = []; test_wordps = []
for i in tqdm(range(math.floor(test_final.shape[0]/batch_size))):
    idx = i * batch_size
    words, wordps = wordp_han_preprocess(test_final.content.values[idx:idx + batch_size])
    test_words.append(words); test_wordps.append(wordps); gc.collect()

test_final_word_han_seq = np.concatenate(test_words)
test_final_wordp_han_seq = np.concatenate(test_wordps)
pickle.dump(test_final_word_han_seq[:200000], open(Config.cache_dir+"/g_test_final_word_han_seq_%s_1.pkl"%(Config.sentence_num*Config.sentence_word_length),"wb"))
pickle.dump(test_final_word_han_seq[200000:], open(Config.cache_dir+"/g_test_final_word_han_seq_%s_2.pkl"%(Config.sentence_num*Config.sentence_word_length),"wb"))
pickle.dump(test_final_wordp_han_seq[:200000], open(Config.cache_dir+"/g_test_final_wordp_han_seq_%s_1.pkl"%(Config.sentence_num*Config.sentence_word_length),"wb"))
pickle.dump(test_final_wordp_han_seq,[200000:] open(Config.cache_dir+"/g_test_final_wordp_han_seq_%s_2.pkl"%(Config.sentence_num*Config.sentence_word_length),"wb"))

test_words = []; test_wordps = []
for i in tqdm(range(math.floor(test_final.shape[0]/batch_size))):
    idx = i * batch_size
    words, wordps = get_wordp_seq(test_final.content.values[idx:idx + batch_size])
    test_words.append(words); test_wordps.append(wordps); gc.collect()

test_final_word_seq = np.concatenate(test_words)
test_final_wordp_seq = np.concatenate(test_wordps)
pickle.dump(test_final_word_seq, open(Config.cache_dir+"/g_test_final_word_seq_%s.pkl"%Config.word_seq_maxlen,"wb"))
pickle.dump(test_final_wordp_seq, open(Config.cache_dir+"/g_test_final_wordp_seq_%s.pkl"%Config.word_seq_maxlen,"wb"))
