# ハイパーパラメータ等を入力し、クラス分類(訓練、検証)
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim as optimizer
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from .classifier_base import _ClassifierBase

'''
numpy.ndarray->torch.tensor
data: ndarray
data_type: numpy dtype
           ex. torch.double=np.float64, torch.float=np.float32, torch.long=np.int64, 
device: cpu or cuda:0(auto: cpu if cuda:0 is unavailable)
'''
def np2torch(data, data_type=np.float32, device='auto') :
    if device=='auto' :
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data = torch.from_numpy(data.astype(data_type)).clone().to(device)
    return data

'''
torch.tensor->numpy.ndarray
data: torch.tensor
data_type: numpy dtype
'''
def torch2np(data, data_type=np.float32):
    data = data.to('cpu').detach().numpy().copy().astype(data_type)
    return data

'''
訓練、テスト分割(+ torch.tensorへの変換)
data, label: データ、ラベル(サンプル数は同じ)
train_size: 0~1.0
is_shuffled: シャッフルを有効化
rand_seed: シャッフルシード値
is_torch: torch.tensorへの変換を有効化(ndarrayのみ、ラベルは数値化が必要)
'''
def split_train_test(data, label, train_size=0.75, 
                     is_shuffled=False, rand_seed=None, 
                     cast_torch=False, cast_data_type=np.float32, cast_label_type=np.int64) :
    train_x, test_x, train_y, test_y = train_test_split(data, label, train_size=train_size,
                                                        shuffle=is_shuffled, random_state=rand_seed)
    if cast_torch: 
        train_x = np2torch(train_x, data_type=cast_data_type)
        train_y = np2torch(train_y, data_type=cast_label_type)
        test_x = np2torch(test_x, data_type=cast_data_type)
        test_y = np2torch(test_y, data_type=cast_label_type)
    return train_x, test_x, train_y, test_y

# DNN分類器
class DNNClassifier(_ClassifierBase):
    '''
    model: モデルクラス
    model: モデル引数(辞書型)
    loss_func: 損失関数
    loss_args: 損失関数引数(辞書型)
    optim: 最適化関数
    optim_args: 最適化関数引数(辞書型、model.parameters()以外)
    '''
    def __init__(self, model, model_args={}, 
                 loss_func=nn.CrossEntropyLoss, loss_args={}, 
                 optim=optimizer.Adam, optim_args={}):
        self.model = model(**model_args) # モデル
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.loss_func = loss_func(**loss_args) # 損失関数
        self.optim = optim(self.model.parameters(), **optim_args) # 最適化関数
        self.train_outputs = torch.tensor([], device=self.device) # 各エポックの出力(Epoch x n_data x n_cls)
        self.train_losses = torch.tensor([]) # 各エポックの損失
        self.train_accs = torch.tensor([]) # 各エポックの精度
        self.test_outputs = torch.tensor([], device=self.device) # 各エポックの出力
        self.test_losses = torch.tensor([]) # 各エポックの損失
        self.test_accs = torch.tensor([]) # 各エポックの精度
        
    '''
    訓練
    train_x: 訓練データ(torch.tensor)
    train_y: 訓練ラベル(torch.tensor)
    epoch: エポック数
    batch_size: バッチサイズ
    add_func: モデル出力に追加で適用する関数
    keep_outputs: 出力を何エポックごとに保持するか(データ量を減らす)
    keep_losses: 損失を何エポックごとに保持するか
    keep_accs: 精度を何エポックごとに保持するか
    verbose: 何エポックごとに結果(損失と精度)を表示するか(0:出力無し)
    log_fn: 結果をlogに(None: 標準出力) ＊未実装
    to_np: 結果をnumpyに変換
    '''
    def train(self, train_x, train_y, epoch, batch_size, 
              add_func=None,
              keep_outputs=1, keep_losses=1, keep_accs=1, verbose=1, log_fn=None, to_np=False) :
        # DataLoader
        train_data_size = train_x.size()[0]
        train_ds = TensorDataset(train_x, train_y)
        # shuffleはシード値指定できないから無し or 手動
        train_loader = DataLoader(train_ds, batch_size=batch_size)

        print('Start Training')
        self.model.train()
        for e in range(1, epoch+1):
            print('Epoch: {}'.format(e))
            epoch_outputs = torch.tensor([], device=self.device)
            epoch_loss = 0
            epoch_hit = 0
            for x, y in train_loader :
                # 出力
                pred_y = self.model(x)
                # 追加処理
                if add_func is not None :
                    pred_y = add_func(pred_y)
                epoch_outputs = torch.cat((epoch_outputs, pred_y),dim=0)
                # 勾配の初期化
                self.optim.zero_grad()
                # 損失の計算
                loss = self.loss_func(pred_y, y)
                epoch_loss += loss.item()
                # 勾配の計算(誤差逆伝播) 
                loss.backward()
                # 重みの更新
                self.optim.step()
                # 正解数
                _, pred_class = pred_y.max(dim=1)
                epoch_hit += (pred_class == y).sum().item()
            
            # 結果
            if e%verbose==0 :
                print('Epoch Loss: {}'.format(epoch_loss))
                print('Epoch Acc: {}\n'.format(epoch_hit/train_data_size))
            # 結果保存
            if e%keep_outputs == 0 :
                self.train_outputs = torch.cat((self.train_outputs,epoch_outputs.unsqueeze(dim=0)), dim=0)
            if e%keep_losses == 0 :
                self.train_losses = torch.cat((self.train_losses, torch.tensor([epoch_loss])), dim=0)
            if e%keep_accs==0 :
                epoch_acc = epoch_hit/train_data_size
                self.train_accs = torch.cat((self.train_accs, torch.tensor([epoch_acc])), dim=0)
        # numpyに変換するか
        if to_np :
            return torch2np(self.train_losses), torch2np(self.train_accs)
        else :
            return self.train_losses, self.train_accs

    '''
    テスト
    test_x: テストデータ(torch.tensor)
    test_y: テストラベル(torch.tensor)
    add_func: モデル出力に追加で適用する関数
    keep_outputs: 出力を保持するか(0:無 or 1:有)
    keep_losses: 損失を保持するか(0:無 or 1:有)
    keep_accs: 精度を保持するか(0:無 or 1:有)
    verbose: 結果(損失と精度)を表示するか(0:無 or 1:有)
    log_fn: 結果をlogに(None: 標準出力) ＊未実装
    to_np: 結果をnumpyに変換
    '''
    # テスト
    def test(self, test_x, test_y, add_func=None, 
             keep_outputs=1, keep_losses=1, keep_accs=1, verbose=1, log_fn=None, to_np=False) :
        # DataLoader
        test_data_size = test_x.size()[0]
        test_ds = TensorDataset(test_x, test_y)
        # shuffleはシード値指定できないから無し or 手動
        test_loader = DataLoader(test_ds, batch_size=test_data_size)

        print('Start Test')
        self.model.eval()
        epoch_outputs = torch.tensor([], device=self.device)
        epoch_loss = 0
        epoch_hit = 0
        for x, y in test_loader :
            # 勾配計算をしない場合
            with torch.no_grad() :
                # 出力
                pred_y = self.model(x)
                if add_func is not None :
                    pred_y = add_func(pred_y)
                epoch_outputs = torch.cat((epoch_outputs, pred_y),dim=0)
                # 損失の計算
                loss = self.loss_func(pred_y, y) 
                self.test_losses[0] = loss.item()
                # 正解数
                _, pred_class = pred_y.max(dim=1)
                hit = (pred_class == y).sum().item()
        self.test_accs[0] = hit/test_data_size
        
        # 出力
        if verbose :
            print('Loss: {}'.format(self.test_losses[0].item()))
            print('Acc: {}\n'.format(self.test_accs[0].item()))
        # 結果保存
        if keep_outputs :
            self.test_outputs = torch.cat((self.test_outputs,epoch_outputs.unsqueeze(dim=0)), dim=0)
        if keep_losses :
            self.test_losses = torch.cat((self.test_losses, torch.tensor([epoch_loss])), dim=0)
        if keep_accs:
            epoch_acc = epoch_hit/test_data_size
            self.test_accs = torch.cat((self.test_accs, torch.tensor([epoch_acc])), dim=0)
        # numpyに変換するか
        if to_np :
            return torch2np(self.test_losses), torch2np(self.test_accs)
        else :
            return self.test_losses, self.test_accs

    '''
    エポック毎に訓練+テスト
    train_x: 訓練データ(torch.tensor)
    train_y: 訓練ラベル(torch.tensor)
    test_x: テストデータ(torch.tensor)
    test_y: テストラベル(torch.tensor)
    epoch: エポック数
    batch_size: バッチサイズ
    add_func: モデル出力に追加で適用する関数
    keep_outputs: 出力を何エポックごとに保持するか(データ量を減らす)
    keep_losses: 損失を何エポックごとに保持するか
    keep_accs: 精度を何エポックごとに保持するか
    verbose: 何エポックごとに結果(損失と精度)を表示するか(0:出力無し)
    log_fn: 結果をlogに(None: 標準出力) ＊未実装
    to_np: 結果をnumpyに変換
    '''
    def train_test(self, train_x, train_y, test_x, test_y, epoch, batch_size, add_func=None,
                   keep_outputs=1, keep_losses=1, keep_accs=1, verbose=1, log_fn=None, to_np=False) :
        # DataLoader
        train_data_size = train_x.size()[0]
        test_data_size = test_x.size()[0]
        train_ds = TensorDataset(train_x, train_y)
        test_ds = TensorDataset(test_x, test_y)
        # shuffleはシード値指定できないから無し or 手動
        train_loader = DataLoader(train_ds, batch_size=batch_size)
        test_loader = DataLoader(test_ds, batch_size=test_data_size)

        print('Start Training & Test')
        
        for e in range(1, epoch+1):
            print('Epoch: {}'.format(e))
            # 訓練
            self.model.train()
            epoch_outputs = torch.tensor([], device=self.device)
            epoch_loss = 0
            epoch_hit = 0
            for x, y in train_loader :
                # 出力
                pred_y = self.model(x)
                if add_func is not None :
                    pred_y = add_func(pred_y)
                epoch_outputs = torch.cat((epoch_outputs, pred_y),dim=0)
                # 勾配の初期化
                self.optim.zero_grad()
                # 損失の計算
                loss = self.loss_func(pred_y, y) 
                epoch_loss += loss.item()
                # 勾配の計算(誤差逆伝播) 
                loss.backward()
                # 重みの更新
                self.optim.step()
                # 正解数
                _, pred_class = pred_y.max(dim=1)
                epoch_hit += (pred_class == y).sum().item()
            # 結果
            if e%verbose==0 :
                print('Training')
                print('Epoch Loss: {}'.format(epoch_loss))
                print('Epoch Acc: {}'.format(epoch_hit/train_data_size))
            # 結果保存
            if e%keep_outputs == 0 :
                self.train_outputs = torch.cat((self.train_outputs,epoch_outputs.unsqueeze(dim=0)), dim=0)
            if e%keep_losses == 0 :
                self.train_losses = torch.cat((self.train_losses, torch.tensor([epoch_loss])), dim=0)
            if e%keep_accs==0 :
                epoch_acc = epoch_hit/train_data_size
                self.train_accs = torch.cat((self.train_accs, torch.tensor([epoch_acc])), dim=0)
            
            # テスト
            self.model.eval()
            epoch_outputs = torch.tensor([], device=self.device)
            epoch_loss = 0
            epoch_hit = 0
            for x, y in test_loader :
                # 勾配計算をしない場合
                with torch.no_grad() :
                    # 出力
                    pred_y = self.model(x)
                    if add_func is not None :
                        pred_y = add_func(pred_y)
                    epoch_outputs = torch.cat((epoch_outputs, pred_y),dim=0)
                    # 損失の計算
                    loss = self.loss_func(pred_y, y) 
                    epoch_loss += loss.item()
                    # 正解数
                    _, pred_class = pred_y.max(dim=1)
                    epoch_hit += (pred_class == y).sum().item()
            # 結果
            if e%verbose==0 :
                print('Test')
                print('Epoch Loss: {}'.format(epoch_loss))
                print('Epoch Acc: {}\n'.format(epoch_hit/test_data_size))
            # 結果保存
            if e%keep_outputs == 0 :
                self.test_outputs = torch.cat((self.test_outputs,epoch_outputs.unsqueeze(dim=0)), dim=0)
            if e%keep_losses == 0 :
                self.test_losses = torch.cat((self.test_losses, torch.tensor([epoch_loss])), dim=0)
            if e%keep_accs==0:
                epoch_acc = epoch_hit/test_data_size
                self.test_accs = torch.cat((self.test_accs, torch.tensor([epoch_acc])), dim=0)
        # numpyに変換するか
        if to_np :
            return torch2np(self.train_losses), torch2np(self.train_accs), \
                   torch2np(self.test_losses), torch2np(self.test_accs) 
        else :
            return self.train_losses, self.train_accs, \
                   self.test_losses, self.test_accs
    
    # モデルのパラメータ保存
    def save_model(self, model_fn) -> None:
        torch.save(self.model.state_dict(), model_fn)

    # モデルのパラメータ読み込み
    # 要確認 パラメータをロードした後、optim(model.parameters())を再生成する必要はないのか
    def load_model(self, model_fn) -> None:
        self.model.load_state_dict(torch.load(model_fn))

    # 出力がうまくできているか(途中)
    def check_outputs(self, is_test=True, log_fn=None) :
        # 何を出力するか
        # Epoch 1
        #   Eval1
        #   Eval2
        # Epoch 2
        
        # Eval
        # 出力をクラスに直して、各クラスの数
        # 値の最大、最小、平均値
        if is_test & (self.test_outputs is not None):
            for e, epoch_outputs in enumerate(self.test_outputs) :
                print('Epoch {}'.format(e))
                print(epoch_outputs)
                # 
            
        if (not is_test) & (self.train_outputs is not None) :
            
            pass
        else :
            print('Outputs is None!')
            print('Do train, test or train_test!')








