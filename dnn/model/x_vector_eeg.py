# X-VectorをEEG用に改変
# ソースコード元: https://github.com/KrishnaDN/x-vector-pytorch/tree/master/models
# 手法(値や層数等): https://www.researchgate.net/publication/332332290_Subspace_techniques_for_task_independent_EEG_person_identification

import torch.nn as nn
import torch.nn.functional as F
import torch
from .my_layers import TimeDistributed, CustomLayer

class TDNN(nn.Module):
    '''
    TimeDelayNeuralNetwork
    n_batch x n_ch x seq_len x n_dataに対応
    '''
    def __init__(self, 
                 input_dim=23, 
                 output_dim=512,
                 context_size=5,
                 stride=1,
                 dilation=1,
                 batch_norm=False,
                 dropout_p=0.2):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

        Affine transformation not applied globally to all frames but smaller windows with local context

        batch_norm: True to include batch normalisation after the non linearity
        
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
      
        self.kernel = nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''
        
        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)
        # Unfold input into smaller temporal contexts
        x = F.unfold(
                        x, 
                        (self.context_size, self.input_dim), 
                        stride=(1,self.input_dim), 
                        dilation=(self.dilation,1)
                    )
        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1,2)
        x = self.kernel(x.float())
        x = self.nonlinearity(x)
        
        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1,2)
            x = self.bn(x)
            x = x.transpose(1,2)

        return x

class X_Vector(nn.Module):
    '''
    X-VectorをEEG用に改変
    '''
    def __init__(self, 
                 n_classes,
                 n_ch,
                 input_dim, 
                 n_tdnn=2,
                 out_tdnn=(512,256),
                 context_sizes=(5,3),
                 n_segment=1,
                 out_segment=(100,),
                 out='pred',
                 size_check=False):
        '''
        n_classes: 分類ラベル数
        n_ch: 電極数
        input_dim: data.shape[-1]
        n_tdnn: tdnn数
        out_tdnn: tdnn出力次元(要素数n_tdnnのタプル)
        context_sizes: tdnnのcontext_size(要素数n_tdnnのタプル)
        n_segment: segment数
        out_segment: segmentの出力次元(要素数n_segmentのタプル)
        out: 最終出力が予測結果orベクトル
        size_check: サイズ確認用
        '''
        super().__init__()
        self.layers = {}
        self.layers['dimshuffle'] =  CustomLayer(lambda x : x.permute(0,2,1,3)) # 次元1,2を入れ替え
        # Frame Level
        # TimeDistributedにより電極毎にTDNN
        for n in range(n_tdnn) :
            self.layers['tdnn{}'.format(n+1)] = TimeDistributed(TDNN(input_dim=input_dim, output_dim=out_tdnn[n], 
                                                        context_size=context_sizes[n], dilation=1,dropout_p=0.5))
            input_dim = out_tdnn[n]
        # Frame levelPooling
        self.layers['stat'] = CustomLayer(lambda x :torch.cat((torch.mean(x,2),torch.var(x,2)),2)) #
        self.layers['flatten'] = nn.Flatten() # nn.FlattenではBatchはそのまま
        # Segment Level
        input_dim = n_ch*input_dim*2 
        for n in range(n_segment) :
            self.layers['segment{}'.format(n+1)] = nn.Linear(input_dim, out_segment[n])
            input_dim = out_segment[n]
        self.layers['output'] = nn.Linear(input_dim, n_classes)
        self.layers['softmax'] = nn.Softmax(dim=1)
        self.layers = nn.ModuleDict(self.layers)
        self.out = out
        self.size_check = size_check
    
    def forward(self, x):
        '''
        分類結果 or x-vectorを返す
        x: n_batch x seq_len x n_ch x n_feat 
        '''
        for name, layer in self.layers.items() :
            x = layer(x)
            if 'segment' in name :
                vec = x # x-vectorを保持
            if self.size_check :
                print('{}:{}'.format(name, x.size()))
        # 最終出力が予測結果 or x-vector
        if self.out=='vec' :
            return vec # batch x out_segment[-1]
        else :
            return x # batch x num_classes

if __name__=='__main__' :
    import torch
    torch.manual_seed(1)
    # Data
    batch_size = 100
    n_ch = 14
    seq_len = 10 # 
    n_data = 128 #
    data = torch.rand(batch_size, seq_len, n_ch, n_data) 
    print('Data: {}'.format(data.shape))
    # Model
    input_dim = data.shape[-1] 
    n_classes = 10
    n_ch = 14
    xv = X_Vector(n_classes,
                  n_ch,
                  input_dim,
                  n_tdnn=2,
                  out_tdnn=(512,256),
                  context_sizes=(5,3),
                  n_segment=1,
                  out_segment=(100,),
                  size_check=True)
    pred, vec = xv(data)
    print('Pred: {}'.format(pred.shape)) # batch x num_classes
    print('Vec: {}'.format(vec.shape)) # batch x out_segment[-1]
