import copy
import torch
import torch.nn as nn

class JointModel(nn.Module) :
    '''
    2つのモデルを結合
    '''
    def __init__(self, main, sub, joint_dim=-1, joint_func=None, train_mode='double', stream='parallel') :
        '''
        main: 最終出力を行うモデル(インスタンス)
        sub: 出力がメインへの入力に結合されるモデル(インスタンス)
        joint_dim: サブ出力と入力の結合次元(-1でAuto, joint_funcを指定した場合使用しない)
        joint_func: サブの出力と入力の結合を行う関数(Noneの場合, torch.cat((x,subout),joint_dim))
        train_mode: 'double':両方, 'main':メインのみ, 'sub':サブのみ 
        stream: 'parallel'= x->[sub]->subout x+subout->[main]->mainout
                'series'=   x=>[sub]->subout   subout->[main]->mainout
        '''
        super().__init__()
        self.main = copy.deepcopy(main)
        self.sub = copy.deepcopy(sub)
        self.joint_dim = joint_dim 
        self.joint_func = joint_func
        self.train_mode = train_mode
        self.stream = stream
        
    def forward(self, x) :
        # サブ
        subout = self.sub(x)
        if self.stream=='parallel':
            # 入力結合
            if self.joint_func is not None :
                x = self.joint_func(x, subout)
            else :
                dim = x.dim()-1 if self.joint_dim==-1 else self.joint_dim
                x = torch.cat((x,subout), dim)
            # メイン
            mainout = self.main(x)
        elif self.stream=='series' :
            mainout = self.main(subout)
        else :
            raise ValueError('stream is "parallel" or "series"')
        return mainout

    def train(self, mode=True) :
        if mode :
            if self.train_mode=='main' :
                self.main.train()
                self.sub.eval()
            elif self.train_mode=='sub' :
                self.main.eval()
                self.sub.train()
            else :
                super().train(mode) # 両方
        else :
            super().train(mode) # 評価モード