# Sklearn MLモデルで分類
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

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


    def train_test(self, train_x, train_y, test_x, test_y, return_type='acc') :
        '''
        訓練+テスト
        train_x: 訓練データ(numpy.ndarray)
        train_y: 訓練ラベル(numpy.ndarray)
        test_x: テストデータ(numpy.ndarray)
        test_y: テストラベル(numpy.ndarray)
        return_type: 'acc', 'conf', 'conf_dict','score'
        '''
        self.model.fit(train_x, train_y)
        self.test_output = self.model.predict(test_x) # 予測結果
        self.conf_mat = confusion_matrix(test_y, self.test_outputs) # 精度
        self.conf_dict = {'TN':self.conf_mat[0,0], 'FP':self.conf_mat[0,1], 'FN':self.conf_mat[1,0], 'TP':self.conf_mat[1,1] }
        acc = (self.conf_dict['TP']+self.conf_dict['TN']) / np.sum(self.conf_mat) # 正解率 TP+TN/ALL
        prec = self.conf_dict['TP'] / (self.conf_dict['TP']+self.conf_dict['FP']) # 適合率 TP/(TP+FP)
        rec = self.conf_dict['TP'] / (self.conf_dict['TP']+self.conf_dict['FN'])# 再現率 TP/(TP+FN)
        spec = self.conf_dict['TN'] / (self.conf_dict['FP']+self.conf_dict['TN'])# 特異率 TN/(FP+TN)
        self.score = {'acc':acc,'prec':prec,'rec':rec,'spec':spec}
        if return_type is 'score':
            return self.score
        elif return_type is 'conf_dict' :
            return self.conf_dict
        elif return_type is 'conf' :
            return  self.conf_mat
        else :
            return acc
        
        
if __name__=='__main__' :
    import sklearn.svm.SVC
    

