# CNN-GRU(https://arxiv.org/pdf/1807.03147.pdf)を再現
# * https://github.com/IoBT-VISTEC/Deep-Learning-for-EEG-Based-Biometrics/blob/master/DEAP-PI.py
#   のKerasのコードをTorchに(デフォルト値は極力Kerasに合わせる)
import numpy as np
import torch.nn as nn
import scipy.stats as stats

# 電極名:(y,x) 左上が(0,0)
ch_map = {'FP1':(0,3), 'AF3':(1,3), 'F7':(2,0), 'F3':(2,2), 'FC1':(3,3), 'FC5':(3,1), 'T7':(4,0), 'C3':(4,2), 
          'CP1':(5,3), 'CP5':(5,1), 'P7':(6,0), 'P3':(6,2), 'PZ':(6,4), 'PO3':(7,3), 'O1':(8,3), 'OZ':(8,4), 
          'O2':(8,5), 'PO4':(7,5), 'P4':(6,6), 'P8':(6,8),'CP6':(5,7), 'CP2':(5,5), 'C4':(4,6), 'T8':(4,8), 
          'FC6':(3,7), 'FC2':(3,5), 'F4':(2,6), 'F8':(2,8), 'AF4':(1,5), 'FP2':(0,5), 'FZ':(2,4), 'CZ':(4,4)}

def channels_mapping(data, len_seq=10, ch_names=list(ch_map.keys()), sfreq=128, mesh_size=9) :
        '''
        時間単位でメッシュに32電極をマッピング(n_sample x len_seq x sfreq x mesh_size x mesh_size)
        data: n_sample x n_ch x n_data
        len_seq: 時間長(min=1, max=n_data/sfreq)
        ch_names: 電極名リスト
        sfreq: サンプリング周波数
        mesh_size: メッシュ一辺
        '''
        n_sample = data.shape[0]
        # メッシュ単位の標準化(=ch単位)
        data = stats.zscore(data, axis=1)
        # マッピング
        meshes = np.zeros((n_sample, len_seq, sfreq, mesh_size, mesh_size)) # 論文と次元の順番が違う(Keras<->Torch) 
        for n, ch in enumerate(ch_names) :
            x = ch_map[ch][1]
            y = ch_map[ch][0]
            for l_s in range(len_seq) :
                meshes[:,l_s,:,y,x] = data[:,n,sfreq*l_s : sfreq*(l_s+1)]
        return meshes

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

# 電極名:(y,x) 左上が(0,0)
ch_map = {'FP1':(0,3), 'AF3':(1,3), 'F7':(2,0), 'F3':(2,2), 'FC1':(3,3), 'FC5':(3,1), 'T7':(4,0), 'C3':(4,2), 
          'CP1':(5,3), 'CP5':(5,1), 'P7':(6,0), 'P3':(6,2), 'PZ':(6,4), 'PO3':(7,3), 'O1':(8,3), 'OZ':(8,4), 
          'O2':(8,5), 'PO4':(7,5), 'P4':(6,6), 'P8':(6,8),'CP6':(5,7), 'CP2':(5,5), 'C4':(4,6), 'T8':(4,8), 
          'FC6':(3,7), 'FC2':(3,5), 'F4':(2,6), 'F8':(2,8), 'AF4':(1,5), 'FP2':(0,5), 'FZ':(2,4), 'CZ':(4,4)}


class CNN_GRU(nn.Module) :
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

# Test
if __name__=='__main__' :
    import torch
    n_sample = 100
    ch_names = list({'FP1':(0,3), 'AF3':(1,3), 'F7':(2,0), 'F3':(2,2), 'FC1':(3,3), 'FC5':(3,1), 'T7':(4,0), 'C3':(4,2), 
              'CP1':(5,3), 'CP5':(5,1), 'P7':(6,0), 'P3':(6,2), 'PZ':(6,4), 'PO3':(7,3), 'O1':(8,3), 'OZ':(8,4), 
              'O2':(8,5), 'PO4':(7,5), 'P4':(6,6), 'P8':(6,8),'CP6':(5,7), 'CP2':(5,5), 'C4':(4,6), 'T8':(4,8), 
              'FC6':(3,7), 'FC2':(3,5), 'F4':(2,6), 'F8':(2,8), 'AF4':(1,5), 'FP2':(0,5), 'FZ':(2,4), 'CZ':(4,4)}.keys())
    n_ch = len(ch_names)
    sfreq = 128
    len_seq = 5
    data = torch.rand(n_sample, n_ch, sfreq*len_seq)
    
    cnn_gru = CNN_GRU(len_seq=len_seq, sfreq=sfreq, ch_names=ch_names, size_check=True)
    x = cnn_gru(data)
    print(x.size())
    # print('0,9,0={}'.format(out[0,9,0]))
    # print(h.size())
    # print('4,0,0={}'.format(h[4,0,0]))

