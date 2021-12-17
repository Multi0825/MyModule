import torch
import torch.nn as nn
import torch.optim as optimizer

class _TrainerBase() :
    '''
    モデル訓練用基底クラス
    '''
    def __init__(self, model, model_args={}, 
                 loss_func=nn.CrossEntropyLoss, loss_args={}, 
                 optim=optimizer.Adam, optim_args={}, init_seed=None, device='cuda:0') :
        '''
        model: モデルクラス
        model_args: モデル引数(辞書型)
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
    
    def __del__(self) :
        '''
        デストラクタ
        GPUを解放できるように
        '''
        del self.model
        torch.cuda.empty_cache() 
    
    def _early_stopping(self, epoch, loss, tolerance_loss=0, tolerance_e=0) :
        '''
        Early Stopping
        |loss(e)| - |loss(e-1)|がtolerance_loss超の場合がtolerance_e以上続いたときにTrue
        デフォルト: エポック間のlossが増加した時点で終了
        epoch: 現在のエポック
        loss: 現在のロス
        tolerance_loss: ロスの増加許容範囲
        patience_loss: ロス増加時からエポックの許容範囲
        '''
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

    def train(self) :
        pass
    def test(self) :
        pass
    def train_test(self) :
        pass

    def save_model(self, model_fn) -> None:
        '''
        モデル保存
        '''
        torch.save(self.model.state_dict(), model_fn)

    def load_model(self, model_fn) -> None:
        '''
        モデルロード
        '''
        self.model.load_state_dict(torch.load(model_fn))
    # 特定の引数を抽出
    # @staticmethod
    # def _parse_args(kwargs, group) :
    #     target_keys = [key for key in kwargs.keys() if group in key]
    #     new_kwargs = {key.split('__')[1]:kwargs[key] for key in target_keys}
    #     return new_kwargs