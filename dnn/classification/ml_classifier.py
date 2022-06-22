# Sklearn MLモデルで分類
import os
import numpy as np
from sklearn.metrics import confusion_matrix

class MLClassifier() :   
    '''
    Scikit-learn MLモデル用分類器
    '''
    def __init__(self, model, model_args={}, 
                 init_seed=None) :
        '''
        model: モデルクラス(Scikit-learn MLモデル)
        model_args: モデル引数(辞書型)、sklearnの場合, max_iterやrandom_stateもここで指定
        init_seed: model_args['random_state']でも可(一応dnn_classifierに合わせて)
        '''
        if init_seed is not None :
            model_args['random_state'] = init_seed
        self.model=model(**model_args)
        # テスト
        self.test_outputs = 0 # 出力
        self.test_accs = 0 # 精度


    def train_test(self, train_x, train_y, test_x, test_y, only_acc=True) :
        '''
        訓練+テスト
        train_x: 訓練データ(numpy.ndarray)
        train_y: 訓練ラベル(numpy.ndarray)
        test_x: テストデータ(numpy.ndarray)
        test_y: テストラベル(numpy.ndarray)
        '''
        self.model.fit(train_x, train_y)
        pred = self.tr_model.predict(test_x) # 予測結果
        acc = confusion_matrix(test_y, pred) # 精度

    def conf_mats(self) :
        '''
        混同行列生成(epoch x conf_mat)
        TN(0,0) FP(0,1)
        FN(1,0) TP(1,1)
        train: 訓練結果を対象に(デフォルトはテスト)
        '''
        n_outputs = self.train_labels.size(0) if train else self.test_labels.size(0)
        conf_mats = []
        for no in range(n_outputs) :
            epoch_outputs = self.train_outputs[no] if train else self.test_outputs[no]
            _, out2cls = epoch_outputs.max(dim=1) # 出力をクラスに変換
            epoch_labels = self.train_labels[no].to(torch.int) if train else self.test_labels[no].to(torch.int) 
            c_m = confusion_matrix(epoch_labels, out2cls) # 混同行列
            conf_mats.append(c_m)
        return np.array(conf_mats)
    
    def scores(self, train=False) :
        '''
        分類問題評価指標(Accuracy, Precision, Recall, Specificiy)取得(epoch x conf_mat)
        train: 訓練結果を対象
        '''
        conf_mats = self.conf_mats(train=train)
        names_confs = {'TN':conf_mats[:,0,0], 'FP':conf_mats[:,0,1], 'FN':conf_mats[:,1,0], 'TP':conf_mats[:,1,1] }
        accs = (names_confs['TP']+names_confs['TN']) / np.sum(conf_mats.reshape((conf_mats.shape[0], -1)), axis=1) # 正解率 TP+TN/ALL
        precs = names_confs['TP'] / (names_confs['TP']+names_confs['FP']) # 適合率 TP/(TP+FP)
        recs = names_confs['TP'] / (names_confs['TP']+names_confs['FN'])# 再現率 TP/(TP+FN)
        specs = names_confs['TN'] / (names_confs['FP']+names_confs['TN'])# 特異率 TN/(FP+TN)
        return accs, precs, recs, specs  


