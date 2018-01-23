import sys
sys.path.append("..")
from config import *
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

class Tokenizer():
    def __init__(self):
        self.n = 0
    def __call__(self,line):
        tokens = []
        for query in line.split('\t'):
            words = [word for word in jieba.cut(query)]
            for gram in [1,2]:
                for i in range(len(words) - gram + 1):
                    tokens += ["_*_".join(words[i:i+gram])]
        self.n += 1
        if self.n%10000==0:
            print(self.n)
        return tokens

def get_tfidf_feature(contents, remarks=""):
    result_path = Config.cache_dir + "/tfidf_%s.pkl"%remarks
    if os.path.exists(result_path):
        result = pickle.load(open(result_path,"rb"))
    else:
        tfidf_vec = TfidfVectorizer(tokenizer=Tokenizer(), min_df=3, max_df=0.95, sublinear_tf=True)
        result = tfidf_vec.fit_transform(contents.astype(str))
        print("tfidf shape is ",result.shape,"="*10,"vocabulary counts : ", len(tfidf_vec.vocabulary_))
        pickle.dump(result, open(result_path,"wb"))
    return result
