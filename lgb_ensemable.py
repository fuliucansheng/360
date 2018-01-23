import lightgbm as lgb
from utils.data import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

sub = True
model_list = [
    "char_cnn",
    "word_cnn",
    "word_char_cnn"
]

featfilter = ["label"]
val = pd.read_csv(Config.cache_dir+"/val.csv", sep="\t")

train_feat = None
for i,v in enumerate(model_list):
    model_val_pred = np.load(Config.cache_dir + "/val_%s.pred"%v)
    val_pred_dataframe = pd.DataFrame(model_val_pred, columns=["%s_neg"%v, "%s_pos"%v])
    #del val_pred_dataframe["%s_neg"%v],val_pred_dataframe["%s_pos"%v]
    if i == 0:
        train_feat = val_pred_dataframe
    else:
        train_feat = pd.concat([train_feat, val_pred_dataframe], axis = 1)

train_feat["label"] = val.label
featfilter = ["label"]
predictors = [i for i in train_feat.columns if i not in featfilter]
print(train_feat.columns)

test_feat = None
for i,v in enumerate(model_list):
    model_test_pred = np.load(Config.cache_dir + "/test_final_%s.pred"%v)
    test_pred_dataframe = pd.DataFrame(model_test_pred, columns=["%s_neg"%v, "%s_pos"%v])
    #del test_pred_dataframe["%s_neg"%v],test_pred_dataframe["%s_pos"%v]
    if i == 0:
        test_feat = test_pred_dataframe
    else:
        test_feat = pd.concat([test_feat, test_pred_dataframe], axis = 1)

## LGB Model Train
params = {
    "boosting":"gbdt",
    "num_leaves": 15,
    "objective": "binary",
    "learning_rate": 0.001,
    #"feature_fraction": 0.886,
    #"bagging_fraction": 0.886,
    #"bagging_freq": 5,
    "is_unbalance": True,
    "min_data_in_leaf": 50,
    "metric": ["auc"]
}
num_round = 3600

x_,y_ = train_test_split(train_feat, test_size=0.1, random_state=0)
x = lgb.Dataset(x_[predictors],label = x_['label'])
y = lgb.Dataset(y_[predictors],label = y_['label'])

def f1(p, d):
    l = d.get_label()
    return "f1_score", f1_score(l, list(map(int, p>0.5))), True
gbm = lgb.train(params, x, num_round, feval = f1, early_stopping_rounds=20, valid_sets=[x,y])

## Print Feature Importance
names, importances = zip(*(sorted(zip(gbm.feature_name(), gbm.feature_importance()), key=lambda x: x[1])))
for name, importance in zip(names, importances):
    print(name, importance)

## Print f1_score offline
p = gbm.predict(y_[predictors])
p_ = (p>0.5).astype("int")
logging.info("f1_score offline: " + str(f1_score(y.label, p_)))

if sub:
    test = get_test_final_data()
    test['pred'] = gbm.predict(test_feat[predictors].astype(float), num_iteration=gbm.best_iteration)
    test['pred'] = (test['pred'] > 0.5).astype(int)
    submit(test)
