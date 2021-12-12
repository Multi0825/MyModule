# オートエンコーダによる回帰(訓練、検証)
# DNN Regressorに機能を追加
# ・Encoderの途中出力も保存可能！
# ・Encoder＋分類器に変形可能！
# ・分類器のみの学習可能！

import matplotlib.pyplot as plt
import logging
from logging import getLogger, Formatter, StreamHandler, FileHandler
import torch
import torch.nn as nn
from torch import optim as optimizer
from torch.utils.data import TensorDataset, DataLoader
from .dnn_regressor import DNNRegressor
from ..utils import torch2np

class _AutoEncoder(nn.Module) :
    '''
    Regressor用AutoEncoder(最適化関数等を1つにまとめたいが、中間出力等が欲しいため)
    *EncoderとDecoderは引数を1つのみ
    これに合ったクラスを作成する必要有
    '''
    def __init__(self, encoder, decoder) :
        '''
        encoder, decoder: インスタンス
        '''
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x, keep_mid_outputs) :
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        if keep_mid_outputs :
            return x1, x2
        else :
            return x1


# AutoEncoder回帰学習器
class AutoEncoderRegressor(DNNRegressor):
    def __init__(self, encoder, decoder, 
                 encoder_args={}, decoder_args={}, 
                 loss_func=nn.MSELoss, loss_args={}, 
                 optim=optimizer.Adam, optim_args={}, 
                 init_seed=None, device='cuda:0') -> None:
        '''
        encoder: Encoderクラス
        Decoder: Decoderクラス
        encoder_args: Encoder引数(辞書型)
        decoder_args: Decoder引数(辞書型)
        loss_func: 損失関数
        loss_args: 損失関数引数(辞書型)
        optim: 最適化関数
        optim_args: 最適化関数引数(辞書型、model.parameters()以外)
        init_seed: モデルのパラメータの初期化のシード(ただここでシード指定しても、いたる箇所で乱数の影響があるため固定は完全同一は無理)
        device: 使用デバイス
        '''
        if init_seed is not None :
            torch.manual_seed(init_seed)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.mode = 'reg' # reg, cls
        # AutoEncoder
        self.autoencoder = _AutoEncoder(encoder(**encoder_args), decoder(**decoder_args))
        self.loss_func = {'reg': loss_func(**loss_args)} # 損失関数
        self.optim = {'reg': optim(self.autoencoder.parameters(), **optim_args)} # 最適化関数
        # 訓練
        self.epoch_count = {'reg': 0}
        self.train_outputs = {'reg': torch.tensor([])} # 各エポックの出力
        self.train_mid_outputs = {'reg': torch.tensor([])}
        self.train_labels = {'reg': torch.tensor([])} # 各エポックの出力に対応するラベル
        self.train_losses = {'reg': torch.tensor([])} # 各エポックの損失
        # テスト
        self.test_outputs = {'reg': torch.tensor([])} # 各エポックの出力
        self.test_mid_outputs = {'reg': torch.tensor([])}
        self.test_labels = {'reg': torch.tensor([])} # 各エポックの出力に対応するラベル
        self.test_losses = {'reg': torch.tensor([])} # 各エポックの損失        
        # 
        self.classifier = None

    def __del__(self) :
        '''
        デストラクタ
        GPUを解放できるように
        '''
        del self.autoencoder, self.classifier
        torch.cuda.empty_cache() 


    # Encoder
    def to_classifier(self, classifier, classifier_args={}, 
                      loss_func=nn.CrossEntropyLoss, loss_args={}, 
                      optim=optimizer.Adam, optim_args={}, init_seed=None, device='cuda:0') :
        '''
        classifier: 分類器クラス
        classifier_args: 分類器引数
        loss_func: 損失関数
        loss_args: 損失関数引数(辞書型)
        optim: 最適化関数
        optim_args: 最適化関数引数(辞書型、model.parameters()以外)
        init_seed: モデルのパラメータの初期化のシード(ただここでシード指定しても、いたる箇所で乱数の影響があるため固定は完全同一は無理)
        device: 使用デバイス
        '''
        self.mode = 'cls'
        self.classifier = classifier(**classifier_args)
        self.loss_func['cls'] = loss_func
        pass

    # 要追加 中間層の出力を保存、mode違いに対応、encoderとdecoderの接続部分
    def train(self, train_x, train_y=None, epoch=1, batch_size=1, only_classifier=False,
              extra_func=None, early_stopping=False, tol_loss=0, tol_e=0,
              keep_outputs=1, keep_mid_outputs=None, keep_losses=1, verbose=1, to_np=False) :
        '''
        訓練
        train_x: 訓練データ(torch.tensor)
        train_y: 訓練ラベル(torch.tensor)、None: 入力データが正解
        epoch: エポック数
        batch_size: バッチサイズ
        only_classifier: mode='cls'のとき、分類器のみを訓練するか
        extra_func: モデル出力に追加で適用する関数
        early_stopping: Early Stoppingの有無
        tol_loss: _early_stoppingのtolerance_lossと対応
        tol_e: _early_stoppingのtolerance_eと対応
        keep_outputs: 出力を何エポックごとに保持するか(データ量を減らす)
        keep_mid_outputs: 中間出力を何エポックごとに保持するか(データ量を減らす)
        keep_losses: 損失を何エポックごとに保持するか
        verbose: 何エポックごとに結果(損失)を表示するか(0:出力無)
        to_np: 結果をnumpyに変換
        '''
        # DataLoader
        train_data_size = train_x.size()[0]
        # 正解ラベルがなければ、入力を正解とする
        train_ds = TensorDataset(train_x, train_y) if train_y is None else TensorDataset(train_x, train_y)  
        train_loader = DataLoader(train_ds, batch_size=batch_size)
        if self.mode == 'reg' :
            self.autoencoder.encoder.to(self.device)
            self.autoencoder.decoder.to(self.device)
        else :
            self.autoencoder.encoder.to(self.device)
            self.autoencoder.encoder.eval()
            self.classifier.to(self.device)
            self.classifier.eval()
            if only_classifier :
                
        for e in range(1, epoch+1):
            epoch_outputs = torch.tensor([])
            epoch_mid_outputs = torch.tensor([])
            epoch_labels = torch.tensor([])
            epoch_loss = 0
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

            # 結果
            if e%verbose==0 :
                print('Epoch: {}'.format(e))
                print('Epoch Loss: {}'.format(epoch_loss))
            # 結果保存
            if e%keep_outputs == 0 :
                self.train_outputs = torch.cat((self.train_outputs,epoch_outputs.unsqueeze(dim=0)), dim=0)
                self.train_labels = torch.cat((self.train_labels, epoch_labels.unsqueeze(dim=0)), dim=0)
            if (keep_mid_outputs is not None) & (e%keep_mid_outputs==0) :
                self.train_mid_outputs = torch.cat((self.train_mid_outputs,epoch_mid_outputs.unsqueeze(dim=0)), dim=0)
            if e%keep_losses == 0 :
                self.train_losses = torch.cat((self.train_losses, torch.tensor([epoch_loss])), dim=0)

            # Early Stopping 判定
            if early_stopping : 
                if self._early_stopping(epoch, epoch_loss, tolerance_loss=tol_loss, tolerance_e=tol_e) :
                    break
        self.epoch_count += epoch
        # numpyに変換するか
        if to_np :
            return torch2np(self.train_losses)
        else :
            return self.train_losses


    def test(self, test_x, test_y=None, batch_size=1, extra_func=None, 
             keep_outputs=1, keep_mid_outputs=0, keep_losses=1, verbose=1, to_np=False) :
        '''
        テスト
        test_x: テストデータ(torch.tensor)
        test_y: テストラベル(torch.tensor)、None: 入力データが正解
        batch_size: バッチサイズ
        extra_func: モデル出力に追加で適用する関数
        keep_outputs: 出力を保持するか(0:無 or 1:有)
        keep_losses: 損失を保持するか(0:無 or 1:有)
        verbose: 結果(損失)を表示するか(0:無 or 1:有)
        to_np: 結果をnumpyに変換
        '''
        # DataLoader
        test_data_size = test_x.size()[0]
        test_ds = TensorDataset(test_x, test_x) if test_y is None else TensorDataset(test_x, test_y)  
        # shuffleはシード値指定できないから無し or 手動
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        self.model = self.model.to(self.device) # GPU使用の場合、転送
        self.model.eval()        
        epoch_outputs = torch.tensor([])
        epoch_labels = torch.tensor([])
        epoch_loss = 0
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

        # 出力
        if verbose :
            print('Epoch Loss: {}'.format(epoch_loss))
        # 結果保存
        if keep_outputs :
            self.test_outputs = torch.cat((self.test_outputs,epoch_outputs.unsqueeze(dim=0)), dim=0)
            self.test_labels = torch.cat((self.test_labels, epoch_labels.unsqueeze(dim=0)), dim=0)
        if keep_losses :
            self.test_losses = torch.cat((self.test_losses, torch.tensor([epoch_loss])), dim=0)
        
        # numpyに変換するか
        if to_np :
            return torch2np(self.test_losses)
        else :
            return self.test_losses


    def train_test(self, train_x, test_x, train_y=None, test_y=None, 
                   epoch=1, batch_size=1, extra_func=None,
                   early_stopping=False, tol_loss=0, tol_e=0,
                   keep_outputs=1, keeps_mid_outputs=0, keep_losses=1, verbose=1, to_np=False) :
        '''
        エポック毎に訓練+テスト
        train_x: 訓練データ(torch.tensor)
        train_y: 訓練ラベル(torch.tensor)、None: 入力データが正解
        test_x: テストデータ(torch.tensor)
        test_y: テストラベル(torch.tensor)、None: 入力データが正解
        epoch: エポック数
        batch_size: バッチサイズ
        extra_func: モデル出力に追加で適用する関数
        early_stopping: Early Stoppingの有無
        tol_loss: _early_stoppingのtolerance_lossと対応
        tol_e: _early_stoppingのtolerance_eと対応
        keep_outputs: 出力を何エポックごとに保持するか(データ量を減らす)
        keep_losses: 損失を何エポックごとに保持するか
        verbose: 何エポックごとに結果(損失と精度)を表示するか(0:出力無し)
        to_np: 結果をnumpyに変換
        '''
        # DataLoader
        train_data_size = train_x.size()[0]
        test_data_size = test_x.size()[0]
        train_ds = TensorDataset(train_x, train_y) if train_y is None else TensorDataset(train_x, train_y) 
        test_ds = TensorDataset(test_x, test_x) if test_y is None else TensorDataset(test_x, test_y) 
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

            # 結果
            if e%verbose==0 :
                print('Epoch: {}'.format(e))
                print('Training')
                print('Epoch Loss: {}'.format(epoch_loss))
            # 結果保存
            if e%keep_outputs == 0 :
                self.train_outputs = torch.cat((self.train_outputs,epoch_outputs.unsqueeze(dim=0)), dim=0)
                self.train_labels = torch.cat((self.train_labels, epoch_labels.unsqueeze(dim=0)), dim=0)
            if e%keep_losses == 0 :
                self.train_losses = torch.cat((self.train_losses, torch.tensor([epoch_loss])), dim=0)
            
            # テスト
            self.model.eval()
            epoch_outputs = torch.tensor([])
            epoch_labels = torch.tensor([])
            epoch_loss = 0
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
            # 結果
            if e%verbose==0 :
                print('Test')
                print('Epoch Loss: {}'.format(epoch_loss))

            # 結果保存
            if e%keep_outputs == 0 :
                self.test_outputs = torch.cat((self.test_outputs,epoch_outputs.unsqueeze(dim=0)), dim=0)
                self.test_labels = torch.cat((self.test_labels, epoch_labels.unsqueeze(dim=0)), dim=0)
            if e%keep_losses == 0 :
                self.test_losses = torch.cat((self.test_losses, torch.tensor([epoch_loss])), dim=0)

            # Early Stopping 判定
            if early_stopping : 
                if self._early_stopping(epoch, epoch_loss, tolerance_loss=tol_loss, tolerance_e=tol_e) :
                    break
        self.epoch_count += epoch
        # numpyに変換するか
        if to_np :
            return torch2np(self.train_losses), \
                   torch2np(self.test_losses) 
        else :
            return self.train_losses, \
                   self.test_losses
    
    # モデルのパラメータ保存
    def save_model(self, model_fn) -> None:
        torch.save(self.model.state_dict(), model_fn)

    # モデルのパラメータ読み込み
    # 要確認 パラメータをロードした後、optim(model.parameters())を再生成する必要はないのか
    def load_model(self, model_fn) -> None:
        self.model.load_state_dict(torch.load(model_fn))

    