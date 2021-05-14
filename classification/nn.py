# オリジナルNN
import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module) :
    # n_in: 入力次元
    # n_out: 出力次元
    # *n_mids: 中間層次元リスト
    def __init__(self, n_in, n_out, *n_mids) :
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        layers = []
        # 中間層を1つ以上定義
        if len(n_mids) > 0 :
            for i, n in enumerate(n_mids) :
                # 1層目
                if i == 0 :
                    layers.append(nn.Linear(n_in, n))
                # 2層目以降
                else :
                    layers.append(nn.Linear(layers[-1].out_features, n))
            # 最終層
            layers.append(nn.Linear(layers[-1].out_features, n_out))
        # 中間層が0
        else :
            layers.append(nn.Linear(n_in, n_out))
        self.layers = nn.ModuleList(layers) # ModuleListに変換が必要

    def forward(self, x) :
        x = x.view(-1, self.n_in)
        for i, layer in enumerate(self.layers):
            if i < len(self.layers)-1 :
                x = F.relu(layer(x.float()))
            else :
                # 出力層のみ活性化関数に入れない(CrossEntropyがSoftMaxを含むため)
                x = layer(x)
        return x.squeeze() # 不要な次元を削除
