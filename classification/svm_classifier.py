# SVM分類器+RVM分類器
import os
import numpy as np
from sklearn.svm import SVC # SVM
# from sklearn.multiclass import OneVsRestClassifier # 一対多分類
from sklearn.model_selection import train_test_split, GridSearchCV # 訓練テスト分割
from sklearn.metrics import accuracy_score # 精度導出

# SVM(version2)
class SvmClassifier() :   
    # kernel : カーネル(‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’)(default:'rbf')
    # C : 正則化パラメータ(default:1)
    # gamma : 'rbf','poly','sigmoid',で使うパラメータ(default:'scale' = 1/(n_features * X.var()) )
    # degree ： 'poly'でのみ使うパラメータ
    # tr_data, tr_label: 訓練データ、訓練ラベル
    # test_data, test_label: テストデータ、テストラベル 
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', degree=3., 
                 tr_data=None, tr_label=None, test_data=None, test_label=None) :
        self.tr_model=SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.tr_data = tr_data 
        self.tr_label = tr_label
        self.test_data = test_data 
        self.test_label = test_label

    # データ設定
    def set_data(self, tr_data, tr_label, test_data, test_label) :
        self.tr_data = tr_data 
        self.tr_label = tr_label
        self.test_data = test_data 
        self.test_label = test_label

    # モデル初期化
    def create_new_model(self, kernel='rbf', C=1.0, gamma='scale', degree=3.) :
        self.tr_model=SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
    
    # 訓練
    def train(self) :        
        if (self.tr_data is not None) | (self.tr_label is not None) :
            self.tr_model.fit(self.tr_data, self.tr_label)
        else :
            raise ValueError('Set tr_data and tr_label')
        
    # 訓練(グリッドサーチ＋CrossValidation)
    # kernel : カーネル(‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’)(default:'rbf')
    # C : 正則化パラメータ(default:1)
    # gamma : 'rbf','poly','sigmoid',で使うパラメータ(default:'scale' = 1/(n_features * X.var()) )
    # degree ： 'poly'でのみ使うパラメータ
    # k_fold : CV分割数
    # verbose : グリッドサーチの過程出力の有無(詳細レベル0,1,2,3)
    def train_gscv(self, kernels=['rbf'], Cs=[1.0], gammas=['scale'], degrees=[3.], k_fold=4, verbose=2) :
        param = {'kernel':kernels,
                'C':Cs,
                'gamma':gammas,
                'degree':degrees}
        gscv = GridSearchCV(SVC(), param, cv=k_fold)
        if (self.tr_data is not None) | (self.tr_label is not None) :
            gscv.fit(self.tr_data, self.tr_label)
        else :
            raise ValueError('Set tr_data and tr_label')
        self.tr_model = gscv.best_estimator_
        self.kernel = self.tr_model.kernel
        self.C = self.tr_model.C
        self.gamma = self.tr_model.gamma
        self.degree = self.tr_model.degree
        print('Best Parameter')
        for param in {'kernel':self.kernel, 'C':self.C, 'gamma':self.gamma, 'degree':self.degree}.items() :
            print('{}:{}'.format(param[0], param[1]))
    
    # 検証
    def test(self) :
        if (self.test_data is not None) | (self.test_label is not None) :
            pred = self.tr_model.predict(self.test_data) # 予測結果
            acc = accuracy_score(self.test_label, pred) # 精度
            # print('Test Accuracy:{}'.format(acc))
            return pred, acc
        else :
            raise ValueError('Set test_data and test_label')

# RVM
# RVM(1.,2.は別々のAPIで2.1,2.2は同一APIの別クラス)
# from skrvm import RVC # 1. https://github.com/JamesRitchie/scikit-rvm
# from sklearn_rvm.em_rvm import EMRVC # 2.1 https://sklearn-rvm.readthedocs.io/en/latest/generated/sklearn_rvm.em_rvm.EMRVR.html
# なきゃないでいい
try :
    from sklearn_rvm.rvc_em import RVC2 # 2.2 上記APIの別パターン(パラメータと) ←これを使用(理由はエラーが少なかったから)

    class RvmClassifier() :
        # tr_data, tr_label: 訓練データ、訓練ラベル
        # test_data, test_label: テストデータ、テストラベル    
        # out_path: 出力ディレクトリ
        def __init__(self, tr_data=None, tr_label=None, test_data=None, test_label=None, out_path=None) :
            self.tr_model=None
            self.params = {'kernel':None, 'degree':0, 'alpha':0, 'beta':0}
            self.tr_data = tr_data 
            self.tr_label = tr_label
            self.test_data = test_data 
            self.test_label = test_label
            self.out_path = out_path # 作成ファイル出力先
            if (self.out_path!=None) and (not os.path.exists(self.out_path)):
                os.makedirs(self.out_path)
                # logging.error(self.out_path+" doesn't exist.")
                # raise FileNotFoundError(self.out_path+" is not found !")

        # 訓練
        # RVCに用意されたパラメータとりあえず全部指定可能に(何が必要かわからないので)
        # kernel,degree, gamma: SVMと同様
        # coef0: poly, sigmoid用
        # 他(n_iter_posterior,max_iter, tol, alpha, alpha_max, threshold_alpha, beta, beta_fixed, compute_score, verbose)
        # 初期値は全部デフォルトをわざわざ明示的に
        # ＊ Found array with 0 sample(s) (shape=(0, 1539)) while ... というエラーが出たら
        # パラメータを変えるとエラーが出る分類タスクが変わった(理由は不明、何かしらやりすぎか？)
        def train(self, n_iter_posterior=50, kernel='rbf', gamma='scale', degree=3, coef0=0.0, max_iter=3000,
                tol=1e-3, alpha=1e-6, alpha_max=1e+9, threshold_alpha=1e+9, beta=1e-6, beta_fixed=False,  
                bias_used=True, compute_score=False, verbose=False) :
            self.tr_model = RVC2(n_iter_posterior=n_iter_posterior, kernel=kernel, degree=degree, gamma=gamma, 
                                coef0=coef0, max_iter=max_iter, tol=tol, alpha=alpha, alpha_max=alpha_max,
                                threshold_alpha=threshold_alpha, beta=beta, beta_fixed=beta_fixed, 
                                bias_used=bias_used, compute_score= compute_score,verbose=verbose)
            if (self.tr_data is not None) | (self.tr_label is not None) :
                self.tr_model.fit(np.array(self.tr_data), np.array(self.tr_label))
            else :
                raise ValueError('Set tr_data and tr_label')
            # 指定パラメータ
            self.params['kernel'] = self.tr_model.kernel
            self.params['degree'] = self.tr_model.degree
            # 自動的に更新されるパラメータ(多分)
            self.params['alpha'] = self.tr_model.alpha_
            self.params['gamma'] = self.tr_model.gamma_
            self.params['beta'] = self.tr_model.beta_
            self.params['mu'] = self.tr_model.mu_
            if self.tr_model.kernel=='linear' :
                self.params['coef'] = self.tr_model.coef_ # Linearの時のみ(それ以外のときに呼ぶとエラー)
            
        # 訓練(グリッドサーチ)RVMではCVは必要ない(らしい)
        # 前述のRVMの引数のうち数値指定はとりあえず全部リストに(複数形も適当)
        def train_gscv(self, n_iter_posteriors=[50], kernels=['rbf'], gammas=['scale'], degrees=[3], coef0s=[0.0], max_iters=[3000],
                    tols=[1e-3], alphas=[1e-6], alpha_maxes=[1e+9], threshold_alphas=[1e+9], betas=[1e-6], beta_fixed=False,  
                    bias_used=True, compute_score=False, verbose=False) :
            param = {'n_iter_posterior':n_iter_posteriors, 'kernel':kernels, 'gamma':gammas, 'degree':degrees,
                    'coef0':coef0s, 'max_iter':max_iters, 'tol':tols, 'alpha':alphas, 'alpha_max':alpha_maxes,
                    'threshold_alpha':threshold_alphas,'beta': betas, 'beta_fixed':beta_fixed,
                    'bias_used':bias_used, 'compute_score':compute_score,'verbose':verbose}
            gscv = GridSearchCV(RVC2(), param, cv=1)
            if (self.tr_data is not None) | (self.tr_label is not None) :
                gscv.fit(self.tr_data, self.tr_label)
            else :
                raise ValueError('Set tr_data and tr_label')
            self.tr_model = gscv.best_estimator_
            # 指定パラメータ
            self.params['kernel'] = self.tr_model.kernel
            self.params['degree'] = self.tr_model.degree
            # 自動的に更新されるパラメータ(多分)
            self.params['alpha'] = self.tr_model.alpha_
            self.params['gamma'] = self.tr_model.gamma_
            self.params['beta'] = self.tr_model.beta_
            self.params['mu'] = self.tr_model.mu_
            if self.tr_model.kernel=='linear' :
                self.params['coef'] = self.tr_model.coef_ # Linearの時のみ(それ以外のときに呼ぶとエラー)
            print('Best Parameter')
            for param in self.params.items() :
                print('{}:{}'.format(param[0], param[1]))
        
        # 検証
        def test(self) :
            if self.tr_model==None :
                raise ValueError('First, Train model')
            pred = self.tr_model.predict(self.test_data) # 予測結果
            acc = accuracy_score(self.test_label, pred) # 精度
            # print('Test Accuracy:{}'.format(acc))
            return pred, acc
except :
    print('RVM is unavailable')


