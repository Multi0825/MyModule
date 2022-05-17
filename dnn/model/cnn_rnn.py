# CNN-GRU(https://arxiv.org/pdf/1807.03147.pdf)を再現
# * https://github.com/IoBT-VISTEC/Deep-Learning-for-EEG-Based-Biometrics/blob/master/DEAP-PI.py
#   のKerasのコードをTorchに(デフォルト値は極力Kerasに合わせる)
import numpy as np
import torch.nn as nn
import scipy.stats as stats

class TimeDistributed(nn.Module) :
    '''
    KerasのTimeDistributedの模倣
    tcapelle Thomas Capelle(https://discuss.pytorch.org/t/timedistributed-cnn/51707/11)
    '''
    def __init__(self, module):
        '''
        module: 対象レイヤー
        '''
        super(TimeDistributed, self).__init__()
        self.module = module
        
    def forward(self, x):
        '''
        x: bs, seq_len, ....
        '''            
        if len(x.size()) <= 2:
            return self.module(x)
        in_shape = x.shape
        bs, seq_len = in_shape[0], in_shape[1]
        x = self.module(x.view(bs*seq_len, *in_shape[2:]))
        return x.view(bs, seq_len, *x.shape[1:])

class CNN_RNN(nn.Module) :
    def __init__(self,
                 n_cls,
                 sfreq,
                 mesh_size=9,
                 filters=(128,64,32), 
                 kernel_size=(3,3),
                 fc1_out=128,
                 rnn_type='gru',
                 units=(32,16),
                 p=0.3,
                 eps=0.001, # Keras Default
                 momentum=0.99, # Keras Default
                 size_check=False
                 ) :
        '''
        CNN-GRU,LSTM(https://arxiv.org/pdf/1807.03147.pdf)を再現
        sfreq: サンプリング周波数
        mesh_size: メッシュ一辺
        filters: (CNN1, CNN2, CNN3)
        kernel_size: kernel size of 3xCNN(共通)
        fc1_out: first fc layers's output size
        rnn_type: gru or lstm
        units: RNN hidden_size
        n_cls: 分類クラス数
        p: dropout
        eps: batchnorm, default is 0.99(keras's default)
        momentum: batchnorm, default is 0.99(keras's default)
        size_check: 出力サイズ確認
        '''
        super().__init__()
        self.sfreq = sfreq 
        self.mesh_size = mesh_size
        self.size_check = size_check

        layers = {}
        # CNN1
        n_cnn = 0
        
        layers['cnn1'] = TimeDistributed(nn.Conv2d(in_channels=self.sfreq, out_channels=filters[n_cnn], 
                                                   kernel_size=kernel_size, stride=1, padding=(kernel_size[0]//2, kernel_size[1]//2)))
        # paddingはkerasのpadding='same'(入力と出力の形状を同じに)に合わせて
        layers['bn1'] = TimeDistributed(nn.BatchNorm2d(num_features=filters[n_cnn], eps=eps, momentum=momentum))
        layers['relu1'] = nn.ReLU()
        layers['dropout1'] = nn.Dropout(p=p)

        # CNN2
        n_cnn = 1
        layers['cnn2'] = TimeDistributed(nn.Conv2d(in_channels=filters[n_cnn-1], out_channels=filters[n_cnn], 
                                                   kernel_size=kernel_size, stride=1, padding=(kernel_size[0]//2, kernel_size[1]//2)))
        layers['bn2'] = TimeDistributed(nn.BatchNorm2d(num_features=filters[n_cnn], eps=eps, momentum=momentum))
        layers['relu2'] = nn.ReLU()
        layers['dropout2'] = nn.Dropout(p=p)

        # CNN3
        n_cnn = 2
        layers['cnn3'] = TimeDistributed(nn.Conv2d(in_channels=filters[n_cnn-1], out_channels=filters[n_cnn], 
                                                   kernel_size=kernel_size, stride=1, padding=(kernel_size[0]//2, kernel_size[1]//2)))
        layers['bn3'] = TimeDistributed(nn.BatchNorm2d(num_features=filters[n_cnn], eps=eps, momentum=momentum))
        layers['relu3'] = nn.ReLU()
        layers['dropout3'] = nn.Dropout(p=p)

        # FC
        layers['flatten'] = TimeDistributed(nn.Flatten())
        in_feats = filters[-1] * self.mesh_size**2
        layers['fc1'] = TimeDistributed(nn.Linear(in_features=in_feats, out_features=fc1_out))
        layers['relu4'] = nn.ReLU()
        layers['dropout4'] = nn.Dropout(p=p)

        # GRU1
        RNN = nn.GRU if rnn_type=='gru' else nn.LSTM
        n_gru=0
        layers['rnn1'] = RNN(input_size=fc1_out, hidden_size=units[n_gru], 
                             num_layers=1, dropout=p, batch_first=True)
        layers['dropout5'] = nn.Dropout(p=p)
        
        # GRU2
        n_gru=1
        layers['rnn2'] = RNN(input_size=units[n_gru-1], hidden_size=units[n_gru], 
                             num_layers=1, dropout=p, batch_first=True)
        # layers['last_seq'] = CustomLayer(lambda x: x[:,-1,x]) # 時系列の最後のみ
        layers['dropout6'] = nn.Dropout(p=p)

        # FC
        layers['fc2'] = nn.Linear(in_features=units[-1], out_features=n_cls)
        layers['softmax'] = nn.Softmax(dim=1)

        self.layers = nn.ModuleDict(layers) 
    

    def forward(self, x) :
        '''
        x: n_batch x len_seq x sfreq x mesh_size x mesh_size
        '''
        for name,layer in self.layers.items() :
            if 'rnn1'==name :
                x,h = layer(x)
            elif 'rnn2'==name : # RNN2層目は時系列の最後のみ
                x, h = layer(x)
                x = x[:,-1,:]
            else :
                x = layer(x)
            if self.size_check :
                print('{}:{}'.format(name, x.size()))
        return x


