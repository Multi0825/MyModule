# Encoderに全結合層を結合した分類器
import torch.nn as nn
import copy

class Encoder_Classifier(nn.Module) :
    def __init__(self, encoder, in_feats, out_feats, only_classifier=True) :
        '''
        encoder: encoderインスタンス
        in_feats: Encoderの出力サイズ(input: batch_size x in_feats)
        out_feats: 分類器の出力サイズ(output: batch_size x out_feats)
        only_classifier: 分類器のみの訓練
        '''
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.encoder = copy.deepcopy(encoder)
        self.linear  = nn.Linear(in_feats, out_feats)
        self.softmax = nn.Softmax()
        self.only_classifier = only_classifier
    
    def forward(self, x) :
        x_encoded = self.encoder(x)
        x_encoded = x_encoded.view(x_encoded.size(0), self.in_feats)
        x = self.softmax(self.linear(x_encoded))
        return x

    # エンコーダーのみ訓練に
    def train(self, mode=True) :
        if mode :
            if self.only_classifier :
                self.linear.train()
                self.softmax.train()
                self.encoder.eval()
            else :
                super().train(mode)
        else :
            super().train(mode)

