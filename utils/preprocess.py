import sys
sys.path.append("..")
from functools import partial
from config import *
from keras.utils import to_categorical
from keras.preprocessing import sequence
from pyltp import Postagger,SentenceSplitter
import jieba
import logging
jieba.setLogLevel(logging.INFO)
jieba.enable_parallel(10)
jieba.load_userdict(Config.data_dir + "/libs/dict.txt.big")

char_embed_dict = pickle.load(open(Config.char_embed_dict_path, "rb"))
word_embed_dict = pickle.load(open(Config.word_embed_dict_path, "rb"))
wordp_embed_dict = pickle.load(open(Config.word_property_embed_dict_path, "rb"))
postags = Postagger()
postags.load(Config.data_dir + "/libs/ltp/pos.model")

char_unknown = len(char_embed_dict.keys()) + 1
word_unknown = len(word_embed_dict.keys()) + 1
wordp_unknown = len(wordp_embed_dict.keys()) + 1

def get_word_seq(contents, word_maxlen=Config.word_seq_maxlen, mode="post", keep=False, verbose=False):
    word_r = []
    contents = "\n".join(contents)
    contents = " ".join(list(jieba.cut(contents))).replace(" \n ","\n")
    contents = [content.split(" ") for content in contents.split("\n")]
    for content in tqdm(contents, disable=(not verbose)):
        if keep:
            word_c = np.array([word_embed_dict[w] if w in word_embed_dict else word_unknown for w in content])
        else:
            word_c = np.array([word_embed_dict[w] for w in content if w in word_embed_dict])
        word_r.append(word_c)
    word_seq = sequence.pad_sequences(word_r, maxlen=word_maxlen, padding=mode, truncating=mode, value=0)
    return word_seq

def get_char_seq(contents, char_maxlen=Config.char_seq_maxlen, mode="post", keep=False, verbose=False):
    char_r = []
    for content in tqdm(contents, disable=(not verbose)):
        if keep:
            char_c = np.array([char_embed_dict[c] if c in char_embed_dict else char_unknown for c in content])
        else:
            char_c = np.array([char_embed_dict[c] for c in content if c in char_embed_dict])
        char_r.append(char_c)
    char_seq = sequence.pad_sequences(char_r, maxlen=char_maxlen, padding=mode, truncating=mode, value=0)
    return char_seq

def get_wordp_seq(contents, word_maxlen=Config.word_seq_maxlen, mode="post", keep=False, verbose=False):
    word_r = [];wordp_r = []
    contents = "\n".join(contents)
    contents = " ".join(list(jieba.cut(contents))).replace(" \n ","\n")
    contents = [content.split(" ") for content in contents.split("\n")]
    for content in tqdm(contents, disable=(not verbose)):
        if keep:
            word_c = np.array([word_embed_dict[w] if w in word_embed_dict else word_unknown for w in content])
            wordp_c = np.array([wordp_embed_dict[wp] if wp in wordp_embed_dict else wordp_unknown for wp in postags.postag(content)])
        else:
            word_c = np.array([word_embed_dict[w] for w in content if w in word_embed_dict])
            wordp_c = np.array([wordp_embed_dict[wp] if wp in wordp_embed_dict else wordp_unknown for idx,wp in enumerate(postags.postag(content)) if content[idx] in word_embed_dict])
        word_r.append(word_c);wordp_r.append(wordp_c)
    word_seq = sequence.pad_sequences(word_r, maxlen=word_maxlen, padding=mode, truncating=mode, value=0)
    wordp_seq = sequence.pad_sequences(wordp_r, maxlen=word_maxlen, padding=mode, truncating=mode, value=0)
    return word_seq, wordp_seq

## batch generator
def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]

def batch_generator(contents, labels, batch_size=128, shuffle=True, keep=False, preprocessfunc=None):
    assert preprocessfunc != None
    sample_size = contents.shape[0]
    index_array = np.arange(sample_size)
    while 1:
        if shuffle:
            np.random.shuffle(index_array)
        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            batch_contents = contents[batch_ids]
            batch_contents = preprocessfunc(batch_contents, keep=keep)
            batch_labels = to_categorical(labels[batch_ids])
            yield (batch_contents,batch_labels)

## word preprocess
def word_cnn_preprocess(contents, word_maxlen=Config.word_seq_maxlen, keep=False):
    word_seq = get_word_seq(contents, word_maxlen=word_maxlen, keep=keep)
    return word_seq

def word_han_preprocess(contents, sentence_num=Config.sentence_num, sentence_length=Config.sentence_word_length, keep=False):
    contents_seq = np.zeros(shape=(len(contents), sentence_num, sentence_length))
    for index,content in enumerate(contents):
        if index >= (len(contents)): break
        sentences = SentenceSplitter.split(content)
        word_seq = get_word_seq(sentences, word_maxlen=sentence_length)
        word_seq = word_seq[:sentence_num]
        contents_seq[index][:len(word_seq)] = word_seq
    return contents_seq

def wordp_han_preprocess(contents, sentence_num=Config.sentence_num, sentence_length=Config.sentence_word_length, keep=False):
    contents_word_seq = np.zeros(shape=(len(contents), sentence_num, sentence_length))
    contents_wordp_seq = np.zeros(shape=(len(contents), sentence_num, sentence_length))
    for index,content in enumerate(contents):
        if index >= (len(contents)): break
        sentences = SentenceSplitter.split(content)
        word_seq, wordp_seq= get_wordp_seq(sentences, word_maxlen=sentence_length)
        word_seq = word_seq[:sentence_num];wordp_seq = wordp_seq[:sentence_num]
        contents_word_seq[index][:len(word_seq)] = word_seq
        contents_wordp_seq[index][:len(wordp_seq)] = wordp_seq
    return [contents_word_seq, contents_wordp_seq]

def word_cnn_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, keep=keep, preprocessfunc=word_cnn_preprocess)

def word_han_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, keep=keep, preprocessfunc=word_han_preprocess)

def wordp_han_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, keep=keep, preprocessfunc=wordp_han_preprocess)

## char preprocess
def char_cnn_preprocess(contents, maxlen = Config.char_seq_maxlen, keep=False):
    char_seq = get_char_seq(contents, char_maxlen=maxlen, keep=keep)
    return char_seq

def char_han_preprocess(contents, sentence_num=Config.sentence_num, sentence_length=Config.sentence_char_length, keep=False):
    contents_seq = np.zeros(shape=(len(contents), sentence_num, sentence_length))
    for index,content in enumerate(contents):
        if index >= (len(contents)): break
        sentences = SentenceSplitter.split(content)
        char_seq = get_char_seq(sentences, char_maxlen=sentence_length)
        char_seq = char_seq[:sentence_num]
        contents_seq[index][:len(char_seq)] = char_seq
    return contents_seq

def char_cnn_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, keep=keep, preprocessfunc=char_cnn_preprocess)

def char_han_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, keep=keep, preprocessfunc=char_han_preprocess)

## word char preprocess
def word_char_cnn_preprocess(contents, word_maxlen=Config.word_seq_maxlen, char_maxlen=Config.char_seq_maxlen, keep=False):
    word_seq = get_word_seq(contents, word_maxlen=word_maxlen, keep=keep)
    char_seq = get_char_seq(contents, char_maxlen=char_maxlen, keep=keep)
    return [word_seq, char_seq]

def word_char_han_preprocess(contents, sentence_num=Config.sentence_num, sentence_word_length=Config.sentence_word_length, sentence_char_length=Config.sentence_char_length, keep=False):
    contents_word_seq = np.zeros(shape=(len(contents), sentence_num, sentence_word_length))
    contents_char_seq = np.zeros(shape=(len(contents), sentence_num, sentence_char_length))
    for index,content in enumerate(contents):
        if index >= (len(contents)): break
        sentences = SentenceSplitter.split(content)
        word_seq = get_word_seq(sentences, word_maxlen=sentence_word_length)
        word_seq = word_seq[:sentence_num]
        char_seq = get_char_seq(sentences, char_maxlen=sentence_char_length)
        char_seq = char_seq[:sentence_num]
        contents_word_seq[index][:len(word_seq)] = word_seq
        contents_char_seq[index][:len(char_seq)] = char_seq
    return [contents_word_seq, contents_char_seq]

def word_char_cnn_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, keep=keep, preprocessfunc=word_char_cnn_preprocess)

def word_char_han_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, keep=keep, preprocessfunc=word_char_han_preprocess)

## wordp char preprocess
def wordp_char_cnn_preprocess(contents, word_maxlen=Config.word_seq_maxlen, char_maxlen=Config.char_seq_maxlen, keep=False):
    char_seq = get_char_seq(contents, char_maxlen=char_maxlen, keep=keep)
    word_seq, wordp_seq = get_wordp_seq(contents, word_maxlen=word_maxlen, keep=keep)
    return [word_seq, wordp_seq, char_seq]

def wordp_char_cnn_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, keep=keep, preprocessfunc=wordp_char_cnn_preprocess)

