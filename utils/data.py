import sys
sys.path.append("..")
from config import *
from sklearn.model_selection import train_test_split
from joblib import Parallel,delayed
import jieba
jieba.load_userdict(Config.data_dir + "/libs/dict.txt.big")

def get_data(data_path, label = True):
    data = []
    data_error_rows = []
    for line in codecs.open(data_path, 'r', 'utf-8'):
        segs = line.strip().split("\t")
        if len(segs) == (4 if label else 3):
            row = {}
            row["id"] = segs[0]
            row["title"] = segs[1]
            row["content"] = segs[2]
            if label: row["label"] = segs[3]
            data.append(row)
        elif len(segs) == 2:
            row = {}
            row["id"] = segs[0]
            row["title"] = segs[1]
            row["content"] = ""
            if label: row["label"] = segs[2]
            data.append(row)
        else:
            data_error_rows.append(line)
    data = pd.DataFrame(data)
    if label:
        data = data[["id","title","content","label"]]
    else:
        data = data[["id","title","content"]]
    return data

def get_train_final_data():
    result_path = Config.data_dir + "/train_final.csv"
    if os.path.exists(result_path):
        result = pd.read_csv(result_path,sep="\t")
        result.fillna("",inplace=True)
    else:
        result = get_data(Config.train_final_path)
        result.fillna("",inplace=True)
        result.to_csv(result_path,index=False,sep="\t")
    return result

def get_test_final_data():
    result_path = Config.data_dir + "/test_final.csv"
    if os.path.exists(result_path):
        result = pd.read_csv(result_path,sep="\t")
        result.fillna("",inplace=True)
    else:
        result = get_data(Config.test_final_path, False)
        result.fillna("",inplace=True)
        result.to_csv(result_path,index=False,sep="\t")
    return result

def submit(pred, dump_path=Config.cache_dir+"/submission.csv"):
    pred["pred"] = pred.pred.map(int2label)
    sub = pred[["id", "pred"]]
    sub.to_csv(dump_path, index=False, header=None)
