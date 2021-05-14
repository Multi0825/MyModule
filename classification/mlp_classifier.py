# ハイパーパラメータ等を入力し、クラス分類(訓練、検証)
import copy
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import KFold

# MLP(2クラス分類も多クラスとして扱う)
class MlpClassifier():
    # model: モデルインスタンス
    # tr_data, tr_label: 訓練データ、訓練ラベル
    # test_data, test_label: テストデータ、テストラベル    
    # out_path: 出力ディレクトリ
    def __init__(self, init_model,  out_path=None):
        self.__init_model = copy.deepcopy(init_model) # モデル初期状態
        self.tr_model = [] # 訓練済みモデル
        self.tr_dataset = None # 訓練データセット
        self.test_dataset = None # テストデータセット
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'# GPU有無
        self.__init_model.to(self.device)
        self.out_path = out_path # 作成ファイル出力先
        if (self.out_path!=None) and (not os.path.exists(self.out_path)):
            os.makedirs(self.out_path)
            # raise FileNotFoundError(self.out_path+" is not found !")

    # データを設定
    def set_data(self, tr_data, tr_label, test_data, test_label) :
        # データセットに
        # ndarrayだった場合、特殊な変換
        if(type(tr_data)=='numpy.ndarray'):
            tr_data = torch.from_numpy(tr_data.astype(np.float32)).clone().to(self.device)
            tr_data.requires_grad=True
        else :
            tr_data = torch.tensor(tr_data).to(self.device)
            tr_data.requires_grad=True
        if(type(tr_label)=='numpy.ndarray'):
            tr_label = torch.from_numpy(tr_label.astype(np.float32)).clone().to(self.device)
        else :
            tr_label = torch.tensor(tr_label).to(self.device)
        # n_sample * n_featureに
        if tr_data.dim() > 2:
            tr_data = tr_data.view(tr_data.size(0), -1)
        self.tr_dataset = TensorDataset(tr_data, tr_label)
        # テストも以下同上
        if(type(test_data)=='numpy.ndarray'):
            test_data = torch.from_numpy(test_data.astype(np.float32)).clone().to(self.device)
        else :
            test_data = torch.tensor(test_data).to(self.device)
        if(type(test_label)=='numpy.ndarray'):
            test_label = torch.from_numpy(test_label.astype(np.float32)).clone().to(self.device)
        else :
            test_label = torch.tensor(test_label).to(self.device)
        if test_data.dim() > 2:
            test_data = test_data.view(test_data.size(0), -1)
        self.test_dataset = TensorDataset(test_data, test_label)

    # 訓練
    # epoch: エポック数
    # batch_size: バッチサイズ
    # lr: 学習率
    # k_fold: CV分割数、デフォルト0は訓練、検証データに分けない
    # optim_class: 最適化関数、デフォルトはAdam 
    def train(self, epoch, batch_size, lr=0.01, k_fold=0,\
              optim_class=optim.Adam, lossfunc_class=nn.CrossEntropyLoss):
        if self.tr_dataset == None :
            raise ValueError('No Training Data')

        print('Hyper Parameter')
        print('Epoch: {}\nBatch Size: {}\nLearing Rate: {}\n'.format(epoch, batch_size, lr))
        
        # CVの場合、訓練と検証に分ける   
        if k_fold > 0 :
            print('Cross Validation')
            print('K: ', k_fold)
            is_cv = True
            feature_inds = np.arange(len(self.tr_dataset.tensors[0]))
            kf = KFold(n_splits=k_fold, shuffle=True)
            tr_inds = [] 
            val_inds = []
            for tr_i, val_i in kf.split(feature_inds):
                tr_inds.append(tr_i)
                val_inds.append(val_i)
        # CVでない場合、検証データを作らない
        else :
            k_fold = 1 # ループ回数のため
            is_cv = False
            tr_inds = [np.arange(len(self.tr_dataset.tensors[0]))]
        
        # CVの分割数に応じて初期モデルまたは訓練済みモデルを複製
        # 訓練済みモデルが無ければ、初期モデルをk個複製
        if len(self.tr_model) == 0 :
            model = [copy.deepcopy(self.__init_model) for k in range(k_fold)]
        # 訓練済みモデルが1つならば、それをk個複製
        elif len(self.tr_model) == 1:
            model = [copy.deepcopy(self.tr_model) for k in range(k_fold)]
        # 訓練済みモデルがk個ならば、そのまま代入
        elif len(self.tr_model) == k_fold :
            model = self.tr_model
        # その他はエラー
        else :
            raise ValueError("k_fold's Value Don't Suit tr_model's Length\nk_fold Must Be 1 or tr_model's Length({})".format(len(self.tr_model)))
        
        tr_epoch_losses=np.zeros([epoch, k_fold]) # 訓練エポック平均損失(エポック＊分割数)
        val_epoch_losses=np.zeros([epoch, k_fold]) # 検証エポック平均損失
        tr_epoch_accs=np.zeros([epoch, k_fold]) # 訓練エポック精度
        val_epoch_accs=np.zeros([epoch, k_fold]) # 検証エポック精度

        # 訓練、検証
        print('Start To Train & Validation')
        for k in range(k_fold):
            if is_cv :
                print('K: ',k)
            tr_loader = DataLoader(Subset(self.tr_dataset, tr_inds[k]), batch_size=batch_size, \
                                   shuffle=True, num_workers=0, pin_memory=False) # 訓練ローダ
            if is_cv:
                val_loader = DataLoader(Subset(self.tr_dataset, val_inds[k]), batch_size=batch_size, \
                                        shuffle=True, num_workers=0, pin_memory=False) # 検証ローダ(CV)
            optimizer = optim_class(model[k].parameters(), lr=lr) # オプティマイザ
            lossfunc = lossfunc_class() # 損失関数
            for e in range(epoch):
                print('Epoch: ', e)
                # 訓練
                model[k].train()
                batch_losses = [] # バッチ毎のロス(合計)
                batch_hits = [] # バッチ毎の正解数
                for x, y in tr_loader:
                    y_pred = model[k](x)
                    loss = lossfunc(y_pred, y)
                    # 学習
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # 損失、精度(正解数)導出
                    batch_losses.append(loss.item()*len(x)) # loss.item()はバッチの平均で、そこからバッチ合計を求める
                    _, pred_class = torch.max(y_pred, 1) # 出力値が最大のクラス番号
                    sum_hits = (pred_class==y).sum().item() # 正解数の合計
                    batch_hits.append(sum_hits)
                tr_epoch_losses[e,k] = np.sum(batch_losses)/len(tr_inds[k])
                tr_epoch_accs[e,k] = np.sum(batch_hits)/len(tr_inds[k])

                   
                # 検証(分割交差検証の場合) 
                if is_cv:
                    model[k].eval()
                    batch_losses = [] 
                    batch_hits = [] 
                    with torch.no_grad():
                        for x, y in val_loader:
                            y_pred = model[k](x)
                            # 損失、精度(正解数)導出
                            loss = lossfunc(y_pred, y)
                            batch_losses.append(loss.item()*len(x))
                            _, pred_class = torch.max(y_pred, 1)
                            sum_hits = (pred_class==y).sum().item()
                            batch_hits.append(sum_hits)       
                    val_epoch_losses[e,k] = np.sum(batch_losses)/len(val_inds[k])
                    val_epoch_accs[e,k] = np.sum(batch_hits)/len(val_inds[k])
        # モデル更新
        self.tr_model = model

        if is_cv:
            # 戻り値がtensorだったら修正が必要
            return tr_epoch_losses, tr_epoch_accs, val_epoch_losses, val_epoch_accs
        return tr_epoch_losses, tr_epoch_accs


    # テスト
    # 出力を返す用に修正が必要(なんのための)
    def test(self, lossfunc_class=nn.CrossEntropyLoss) :
        if self.test_dataset == None :
            raise ValueError('No Test Data')
        if len(self.tr_model) == 0 :
            raise ValueError('No Trained Model')
        # 絶対にDataLoader及びDatasetにする必要がない
        test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset), \
                                shuffle=True, num_workers=0, pin_memory=False)
        test_losses = []
        test_accs = []
        test_outs = []
        print('Start To Test')
        lossfunc = lossfunc_class() # 損失関数
        for k in range(len(self.tr_model)) :
            print('Model')
            self.tr_model[k].eval()
            with torch.no_grad():
                for x, y in test_loader:
                    y_pred = self.tr_model[k](x)
                    # 損失、精度導出
                    loss = lossfunc(y_pred, y)
                    test_losses.append(loss.item())
                    _, pred_class = torch.max(y_pred, 1)
                    test_outs.append(pred_class)
                    sum_hits = (pred_class==y).sum().item()
                    test_accs.append(sum_hits/len(x))       
        return test_outs, test_losses, test_accs
    
    def save_model(self, model_fn):
        pass





