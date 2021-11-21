# ハイパーパラメータ等を入力し、クラス分類(訓練、検証)
import random
import numpy as np
import matplotlib.pyplot as plt
import logging
from logging import getLogger, Formatter, StreamHandler, FileHandler
import torch
import torch.nn as nn
from torch import optim as optimizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from .classifier_base import _ClassifierBase

'''
numpy.ndarray->torch.tensor
data: ndarray
data_type: numpy dtype
           ex. torch.double=np.float64, torch.float=np.float32, torch.long=np.int64, 
device: cpu or cuda:0(auto: cpu if cuda:0 is unavailable)
'''
def np2torch(data, data_type=np.float32) :
    data = torch.from_numpy(data.astype(data_type)).clone()
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
cast_data(label)_type: データ(ラベル)の型(npで指定)
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

'''
k-分割交差検証(イテレータ)
data, label: データ、ラベル(サンプル数は同じ)
is_shuffled: シャッフルを有効化
rand_seed: シャッフルシード値
is_torch: torch.tensorへの変換を有効化(ndarrayのみ、ラベルは数値化が必要)
cast_data(label)_type: データ(ラベル)の型(npで指定)
'''
def cross_valid(data, label, k_split, is_shuffled=False, rand_seed=None,
                cast_torch=False, cast_data_type=np.float32, cast_label_type=np.int64) :
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

# DNN分類器
class DNNClassifier(_ClassifierBase):
    '''
    model: モデルクラス
    model: モデル引数(辞書型)
    loss_func: 損失関数
    loss_args: 損失関数引数(辞書型)
    optim: 最適化関数
    optim_args: 最適化関数引数(辞書型、model.parameters()以外)
    init_seed: モデルのパラメータの初期化のシード(ただここでシード指定しても、いたる箇所で乱数の影響があるため固定は完全同一は無理)
    device
    '''
    def __init__(self, model, model_args={}, 
                 loss_func=nn.CrossEntropyLoss, loss_args={}, 
                 optim=optimizer.Adam, optim_args={}, init_seed=None, device='cuda:0') -> None:
        if init_seed is not None :
            torch.manual_seed(init_seed)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = model(**model_args) # モデル
        self.loss_func = loss_func(**loss_args) # 損失関数
        self.optim = optim(self.model.parameters(), **optim_args) # 最適化関数
        # 訓練
        self.epoch_count = 0
        self.train_outputs = torch.tensor([]) # 各エポックの出力(Epoch x n_data x n_cls)
        self.train_labels = torch.tensor([]) # 各エポックの出力に対応するラベル(Epoch x n_data)
        self.train_losses = torch.tensor([]) # 各エポックの損失
        self.train_accs = torch.tensor([]) # 各エポックの精度
        # テスト
        self.test_outputs = torch.tensor([]) # 各エポックの出力
        self.test_labels = torch.tensor([]) # 各エポックの出力に対応するラベル
        self.test_losses = torch.tensor([]) # 各エポックの損失
        self.test_accs = torch.tensor([]) # 各エポックの精度
    '''
    デストラクタ
    
    GPUを解放できるように
    '''
    def __del__(self) :
        del self.model, self.train_outputs, self.train_labels, self.train_losses, self.train_accs,  \
            self.test_outputs, self.test_labels, self.test_losses, self.test_accs
        torch.cuda.empty_cache() 
    '''
    Early Stopping
    |loss(e)| - |loss(e-1)|がtolerance_loss超の場合がtolerance_e以上続いたときにTrue
    epoch: 現在のエポック
    loss: 現在のロス
    tolerance_loss: ロスの増加許容範囲
    patience_loss: ロス増加時からエポックの許容範囲
    '''
    def _early_stopping(self, epoch, loss, tolerance_loss=0, tolerance_e=0) :
        # 負の値は許容しない
        if tolerance_loss < 0 or tolerance_e < 0:
            raise ValueError('tolerance and patience >= 0')
        
        # 初期化
        if epoch==1 :
            self._prev_loss = float('inf') # 過去のロス
            self._end = -1 # ロス増加から数えて、終了のエポック
        
        # ロスの差が許容範囲内、続行
        if (abs(loss)-abs(self._prev_loss)) <= tolerance_loss :
            self._end = -1
            self._prev_loss = loss
            return False
        # 許容範囲外
        else :
            self._prev_loss = loss
            # ロス差範囲外タイミングから終了タイミングを計算
            if self._end == -1 :
                self._end = epoch + tolerance_e
            # 終了タイミングで終了
            if self._end == epoch :
                return True
            # 続行
            else :
                return False              


    '''
    訓練
    train_x: 訓練データ(torch.tensor)
    train_y: 訓練ラベル(torch.tensor)
    epoch: エポック数
    batch_size: バッチサイズ
    extra_func: モデル出力に追加で適用する関数
    early_stopping: Early Stoppingの有無
    tol_loss: _early_stoppingのtolerance_lossと対応
    tol_e: _early_stoppingのtolerance_eと対応
    keep_outputs: 出力を何エポックごとに保持するか(データ量を減らす)
    keep_losses: 損失を何エポックごとに保持するか
    keep_accs: 精度を何エポックごとに保持するか
    verbose: 何エポックごとに結果(損失と精度)を表示するか(0:出力無)
    to_np: 結果をnumpyに変換
    '''
    def train(self, train_x, train_y, epoch, batch_size, 
              extra_func=None, early_stopping=False, tol_loss=0, tol_e=0,
              keep_outputs=1, keep_losses=1, keep_accs=1, verbose=1, to_np=False) :
        # DataLoader
        train_data_size = train_x.size()[0]
        train_ds = TensorDataset(train_x, train_y)
        # shuffleはシード値指定できないから無し or 手動
        train_loader = DataLoader(train_ds, batch_size=batch_size)

        self.model = self.model.to(self.device) # GPU使用の場合、転送
        self.model.train()
        for e in range(1, epoch+1):
            epoch_outputs = torch.tensor([])
            epoch_labels = torch.tensor([])
            epoch_loss = 0
            epoch_hit = 0
            for x, y in train_loader :
                # GPU使用の場合、転送
                x = x.to(self.device)
                y = y.to(self.device)
                # 出力
                pred_y = self.model(x)
                # 追加処理
                if extra_func is not None :
                    pred_y = extra_func(pred_y)
                # 出力、ラベル保存処理
                epoch_outputs = torch.cat((epoch_outputs, pred_y.to('cpu')),dim=0)
                epoch_labels = torch.cat((epoch_labels, y.to('cpu')), dim=0)
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
                print('Epoch: {}'.format(e))
                print('Epoch Loss: {}'.format(epoch_loss))
                print('Epoch Acc: {}'.format(epoch_hit/train_data_size))
            # 結果保存
            if e%keep_outputs == 0 :
                self.train_outputs = torch.cat((self.train_outputs,epoch_outputs.unsqueeze(dim=0)), dim=0)
                self.train_labels = torch.cat((self.train_labels, epoch_labels.unsqueeze(dim=0)), dim=0)
            if e%keep_losses == 0 :
                self.train_losses = torch.cat((self.train_losses, torch.tensor([epoch_loss])), dim=0)
            if e%keep_accs==0 :
                epoch_acc = epoch_hit/train_data_size
                self.train_accs = torch.cat((self.train_accs, torch.tensor([epoch_acc])), dim=0)
            # Early Stopping 判定
            if early_stopping : 
                if self._early_stopping(epoch, epoch_loss, tolerance_loss=tol_loss, tolerance_e=tol_e) :
                    break
        self.epoch_count += epoch
        # numpyに変換するか
        if to_np :
            return torch2np(self.train_losses), torch2np(self.train_accs)
        else :
            return self.train_losses, self.train_accs

    '''
    テスト
    test_x: テストデータ(torch.tensor)
    test_y: テストラベル(torch.tensor)
    batch_size: バッチサイズ
    extra_func: モデル出力に追加で適用する関数
    keep_outputs: 出力を保持するか(0:無 or 1:有)
    keep_losses: 損失を保持するか(0:無 or 1:有)
    keep_accs: 精度を保持するか(0:無 or 1:有)
    verbose: 結果(損失と精度)を表示するか(0:無 or 1:有)
    to_np: 結果をnumpyに変換
    '''
    # テスト
    def test(self, test_x, test_y, batch_size=10, extra_func=None, 
             keep_outputs=1, keep_losses=1, keep_accs=1, verbose=1, to_np=False) :
        # DataLoader
        test_data_size = test_x.size()[0]
        test_ds = TensorDataset(test_x, test_y)
        # shuffleはシード値指定できないから無し or 手動
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        self.model = self.model.to(self.device) # GPU使用の場合、転送
        self.model.eval()        
        epoch_outputs = torch.tensor([])
        epoch_labels = torch.tensor([])
        epoch_loss = 0
        epoch_hit = 0
        for x, y in test_loader :
            # 勾配計算をしない場合
            with torch.no_grad() :
                # GPU使用の場合、転送
                x = x.to(self.device)
                y = y.to(self.device)
                # 出力
                pred_y = self.model(x)
                # 追加処理
                if extra_func is not None :
                    pred_y = extra_func(pred_y)
                # 出力、ラベル保存処理
                epoch_outputs = torch.cat((epoch_outputs, pred_y.to('cpu')),dim=0)
                epoch_labels = torch.cat((epoch_labels, y.to('cpu')), dim=0)
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
            print('Acc: {}'.format(self.test_accs[0].item()))
        # 結果保存
        if keep_outputs :
            self.test_outputs = torch.cat((self.test_outputs,epoch_outputs.unsqueeze(dim=0)), dim=0)
            self.test_labels = torch.cat((self.test_labels, epoch_labels.unsqueeze(dim=0)), dim=0)
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
    extra_func: モデル出力に追加で適用する関数
    early_stopping: Early Stoppingの有無
    tol_loss: _early_stoppingのtolerance_lossと対応
    tol_e: _early_stoppingのtolerance_eと対応
    keep_outputs: 出力を何エポックごとに保持するか(データ量を減らす)
    keep_losses: 損失を何エポックごとに保持するか
    keep_accs: 精度を何エポックごとに保持するか
    verbose: 何エポックごとに結果(損失と精度)を表示するか(0:出力無し)
    log_fn: 結果をlogに(None: 標準出力) ＊未実装
    to_np: 結果をnumpyに変換
    '''
    def train_test(self, train_x, train_y, test_x, test_y, epoch, batch_size, extra_func=None,
                   early_stopping=False, tol_loss=0, tol_e=0,
                   keep_outputs=1, keep_losses=1, keep_accs=1, verbose=1, to_np=False) :
        # DataLoader
        train_data_size = train_x.size()[0]
        test_data_size = test_x.size()[0]
        train_ds = TensorDataset(train_x, train_y)
        test_ds = TensorDataset(test_x, test_y)
        # shuffleはシード値指定できないから無し or 手動
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        # GPU使用の場合、転送
        self.model = self.model.to(self.device) # GPU使用の場合、転送
        for e in range(1, epoch+1):
            # 訓練
            self.model.train()
            epoch_outputs = torch.tensor([])
            epoch_labels = torch.tensor([])
            epoch_loss = 0
            epoch_hit = 0
            for x, y in train_loader :
                # GPU使用の場合、転送
                x = x.to(self.device)
                y = y.to(self.device)
                # 出力
                pred_y = self.model(x)
                # 追加処理
                if extra_func is not None :
                    pred_y = extra_func(pred_y)
                # 出力、ラベル保存処理
                epoch_outputs = torch.cat((epoch_outputs, pred_y.to('cpu')),dim=0)
                epoch_labels = torch.cat((epoch_labels, y.to('cpu')), dim=0)
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
                print('Epoch: {}'.format(e))
                print('Training')
                print('Epoch Loss: {}'.format(epoch_loss))
                print('Epoch Acc: {}'.format(epoch_hit/train_data_size))
            # 結果保存
            if e%keep_outputs == 0 :
                self.train_outputs = torch.cat((self.train_outputs,epoch_outputs.unsqueeze(dim=0)), dim=0)
                self.train_labels = torch.cat((self.train_labels, epoch_labels.unsqueeze(dim=0)), dim=0)
            if e%keep_losses == 0 :
                self.train_losses = torch.cat((self.train_losses, torch.tensor([epoch_loss])), dim=0)
            if e%keep_accs==0 :
                epoch_acc = epoch_hit/train_data_size
                self.train_accs = torch.cat((self.train_accs, torch.tensor([epoch_acc])), dim=0)
            
            # テスト
            self.model.eval()
            epoch_outputs = torch.tensor([])
            epoch_labels = torch.tensor([])
            epoch_loss = 0
            epoch_hit = 0
            for x, y in test_loader :
                # 勾配計算をしない場合
                with torch.no_grad() :
                    # GPU使用の場合、転送
                    x = x.to(self.device)
                    y = y.to(self.device)
                    # 出力
                    pred_y = self.model(x)
                    # 追加処理
                    if extra_func is not None :
                        pred_y = extra_func(pred_y)
                    # 出力、ラベル保存処理
                    epoch_outputs = torch.cat((epoch_outputs, pred_y.to('cpu')),dim=0)
                    epoch_labels = torch.cat((epoch_labels, y.to('cpu')), dim=0)
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
                print('Epoch Acc: {}'.format(epoch_hit/test_data_size))
            # 結果保存
            if e%keep_outputs == 0 :
                self.test_outputs = torch.cat((self.test_outputs,epoch_outputs.unsqueeze(dim=0)), dim=0)
                self.test_labels = torch.cat((self.test_labels, epoch_labels.unsqueeze(dim=0)), dim=0)
            if e%keep_losses == 0 :
                self.test_losses = torch.cat((self.test_losses, torch.tensor([epoch_loss])), dim=0)
            if e%keep_accs==0:
                epoch_acc = epoch_hit/test_data_size
                self.test_accs = torch.cat((self.test_accs, torch.tensor([epoch_acc])), dim=0)
            # Early Stopping 判定
            if early_stopping : 
                if self._early_stopping(epoch, epoch_loss, tolerance_loss=tol_loss, tolerance_e=tol_e) :
                    break
        self.epoch_count += epoch
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

    # テスト結果を確認
    def outputs_test_results(self, log_fn=None, stream=True) :
        logger = getLogger('Test Results')
        logger.setLevel(logging.DEBUG)
        if stream :
            s_handler = StreamHandler()
            s_handler.setLevel(logging.INFO)
            logger.addHandler(s_handler)
        if log_fn is not None :
            f_handler = FileHandler(log_fn, 'w')
            f_handler.setLevel(logging.DEBUG)
            logger.addHandler(f_handler)
            # 時刻とファイル名をファイル出力のみに追加
            f_handler.setFormatter(Formatter('%(asctime)s-%(filename)s')) 
            logger.debug('')
            f_handler.setFormatter(Formatter())
        
        logger.debug('Test Results')
        n_outputs = self.test_labels.size(0)
        for no in range(n_outputs) :
            epoch_outputs = self.test_outputs[no]
            _, out2cls = epoch_outputs.max(dim=1) # 出力をクラスに変換
            epoch_labels = self.test_labels[no].to(torch.int)
            classes = torch.unique(epoch_labels).tolist() # 重複無しクラスリスト
            cls_counts = sorted({c:(out2cls==c).sum().item() for c in classes}.items()) # 出力クラス出現回数
            c_m = confusion_matrix(epoch_labels, out2cls) # 混同行列
            logger.info(\
            'No.{}\n'.format(no) + \
            'Pred :{}\n'.format(out2cls.tolist()) + \
            'True :{}\n'.format(epoch_labels.tolist()) + \
            'Class: {}\n'.format(cls_counts) + \
            'Acc: {}\n'.format(self.test_accs[no].item()) + \
            'Conf Matrix(T\\P): \n' + \
            '{}\n'.format(c_m) + \
            'Model Outputs: \n' + \
            '{}\n'.format(epoch_outputs.tolist()) + \
            'Total Avg: {}\n'.format(epoch_outputs.mean()) + \
            'Class Avg: {}\n'.format([epoch_outputs[:,i].mean().item() for i in range(len(classes))]) + \
            'Total Max: {}\n'.format(epoch_outputs.max()) + \
            'Class Max: {}\n'.format([epoch_outputs[:,i].max().item() for i in range(len(classes))]) + \
            'Total Min: {}\n'.format(epoch_outputs.min()) + \
            'Class Min: {}\n'.format([epoch_outputs[:,i].min().item() for i in range(len(classes))]) + \
            '\n')
        if stream :
            logger.removeHandler(s_handler)
        if log_fn is not None :
            logger.removeHandler(f_handler)
        




        







