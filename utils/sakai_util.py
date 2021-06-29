# 坂井関数(汎用)
import random
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

# データの標準化
# data: dim=2
# axis: 
def standardization(data, axis=0):
    ss = preprocessing.StandardScaler() 
    # fit_transform: データ数*特徴量を特徴量毎(縦)に正規化
    if axis==1 :
        sd_data = ss.fit_transform(data.T).T 
    elif axis==0: 
        sd_data = ss.fit_transform(data)
    else :
        raise ValueError('axis = 0 or 1')
    return sd_data

# データ正規化
# data: n_ch * n_sample
def zscore(data, axis=0) :
    return stats.zscore(data, axis=axis)

# 重複のない乱数生成
# a, b: 範囲
# k: 個数
def rand_ints_nodup(a, b, k):
    ns = []
    while len(ns) < k:
        n = random.randint(a, b)
        if not n in ns:
            ns.append(n)
    return ns

# 分類結果評価
# y_pred: 予測
# y: 正解
def eval_classification(y_pred, y) :
    ac_score  = accuracy_score(y, y_pred) # 正解率
    precision = precision_score(y, y_pred) # 適合率
    recall    = recall_score(y, y_pred) # 再現率
    f1        = f1_score(y, y_pred) # F1値
    kappa     = cohen_kappa_score(y, y_pred) # κ係数
    conf      = confusion_matrix(y, y_pred) # 混同行列
    return ac_score, precision, recall, f1, kappa, conf
