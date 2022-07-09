# dnn関連汎用関数
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

def np2torch(data, data_type=None) :
    '''
    numpy.ndarray->torch.tensor
    data: ndarray
    data_type: numpy dtype.(if setting None, the type is not changed)
               ex. torch.double=np.float64, torch.float=np.float32, torch.long=np.int64, 
    '''
    data = torch.from_numpy(data).clone() if data_type is None \
           else torch.from_numpy(data.astype(data_type)).clone()

    return data


def torch2np(data, data_type=None):
    '''
    torch.tensor->numpy.ndarray
    data: torch.tensor
    data_type: numpy dtype(if setting None, the type is not changed)
    '''
    data = data.to('cpu').detach().numpy().copy() if data_type is None \
           else data.to('cpu').detach().numpy().copy().astype(data_type)
    return data

def flatten(data, first_dim=True) :
    '''
    first_dim: 0次元を平坦化するか
    '''
    if first_dim :
        new_data = torch.flatten(data)
    else :
        f = torch.nn.Flatten()
        new_data = f(data)
    return new_data

def split_train_test(data : np.ndarray, label, train_size=0.75, 
                     is_shuffled=False, rand_seed=None, 
                     cast_torch=False, cast_data_type=np.float32, cast_label_type=np.int64) :
    '''
    訓練、テスト分割(+ torch.tensorへの変換)
    data, label: データ、ラベル(サンプル数は同じ)
    train_size: 訓練データのサイズ、0~1.0の少数で割合、1~n_sampleの整数で直接サンプル数、
                sklearn.train_test_splitでは不可能な0 or 1を指定可
    is_shuffled: シャッフルを有効化
    rand_seed: シャッフルシード値
    cast_torch: torch.tensorへの変換を有効化(ndarrayのみ、ラベルは数値化が必要)
    cast_data(label)_type: データ(ラベル)の型(npで指定),Noneでそのまま
    '''
    if (train_size == 1.0) or (train_size == data.shape[0]) :
        train_x, test_x, train_y, test_y = train_test_split(data, label, train_size=0.9,
                                                            shuffle=is_shuffled, random_state=rand_seed)
        train_x = np.concatenate([train_x, test_x], axis=0)
        train_y = np.concatenate([train_y, test_y], axis=0)
        test_x, test_y = np.array([]), np.array([])
    elif train_size == 0 :
        train_x, test_x, train_y, test_y = train_test_split(data, label, train_size=0.9,
                                                            shuffle=is_shuffled, random_state=rand_seed)
        test_x = np.concatenate([train_x, test_x], axis=0)
        test_y = np.concatenate([train_y, test_y], axis=0)
        train_x, train_y = np.array([]), np.array([])
    else :
        train_x, test_x, train_y, test_y = train_test_split(data, label, train_size=train_size,
                                                            shuffle=is_shuffled, random_state=rand_seed)
    if cast_torch: 
        train_x = np2torch(train_x, data_type=cast_data_type)
        train_y = np2torch(train_y, data_type=cast_label_type)
        test_x = np2torch(test_x, data_type=cast_data_type)
        test_y = np2torch(test_y, data_type=cast_label_type)
    return train_x, test_x, train_y, test_y


def cross_valid(data, label, k_split, is_shuffled=False, rand_seed=None,
                cast_torch=False, cast_data_type=np.float32, cast_label_type=np.int64) :
    '''
    k-分割交差検証(イテレータ)
    data, label: データ、ラベル(サンプル数は同じ)
    is_shuffled: シャッフルを有効化
    rand_seed: シャッフルシード値
    is_torch: torch.tensorへの変換を有効化(ndarrayのみ、ラベルは数値化が必要)
    cast_data(label)_type: データ(ラベル)の型(npで指定)
    '''
    data_size = len(data)
    inds = [i for i in range(data_size)]
    # シャッフル
    if is_shuffled :
        random.seed(rand_seed)
        random.shuffle(inds)
    # データをセット0~k-1に分け、セット1つをテストデータ、その他を訓練にして返す
    # それを全セットがテストデータになるまで繰り返す        
    set_size = data_size // k_split # 商
    rem_size = data_size % k_split # 余り    
    for k in range(k_split) :
        rem = 0 # 均等に割った余り、あれば前のセットからサイズ＋１
        if rem_size > 0 :
            rem = 1
            rem_size -=1
        start = 0 if k == 0 else end
        end = start + set_size + rem 
        test_inds = inds[start:end]
        train_inds = [i for i in inds if i not in test_inds]
        train_x = data[train_inds]
        test_x = data[test_inds]
        train_y = label[train_inds]
        test_y = label[test_inds]
        # torch.tensor変換
        if cast_torch: 
            train_x = np2torch(train_x, data_type=cast_data_type)
            train_y = np2torch(train_y, data_type=cast_label_type)
            test_x = np2torch(test_x, data_type=cast_data_type)
            test_y = np2torch(test_y, data_type=cast_label_type)
        yield train_x, test_x, train_y, test_y

class CustomLayer(nn.Module) :
    '''
    任意の計算が可能な層
    '''
    def __init__(self, func, *args, **kwargs):
        '''
        func: 関数
        args: 可変長引数
        kwargs: 可変長キーワード引数
        '''
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def forward(self, x) :
        x = self.func(x, *self.args, **self.kwargs)
        return x

def eval_classification(y_pred, y) :
    '''
    分類評価指標
    y_pred: 予測
    y: 正解
    '''
    ac_score  = accuracy_score(y, y_pred) # 正解率
    precision = precision_score(y, y_pred) # 適合率
    recall    = recall_score(y, y_pred) # 再現率
    f1        = f1_score(y, y_pred) # F1値
    kappa     = cohen_kappa_score(y, y_pred) # κ係数
    conf      = confusion_matrix(y, y_pred) # 混同行列
    return ac_score, precision, recall, f1, kappa, conf

