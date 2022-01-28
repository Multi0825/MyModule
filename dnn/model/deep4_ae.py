# Deep4NetをAutoEncoderに改良(修士中間で使用)
# 層数を1,1.5,2,3,4で変更可能
# いくつかの引数は削除

import torch.nn as nn
# import modules_d4n_remaked as bd # BraindecodeDeep4Netで使われてたクラス、関数をまとめた
from modules_d4n_remaked import Ensure4d, Expression, AvgPool2dWithConv, \
                                 transpose_time_to_spat, squeeze_final_output, np_to_th


#########################################################################################################
# Encoder
class Deep4Encoder(nn.Module) :
    '''
    Deep4Encoder
    ・Deep4Netデフォルトと流れは同じ
    ・分類器削除
    ・デフォルトでない分岐を削除、使わない引数削除
    when using encoder only, call single() or switch 'is_single' True
    '''
    def __init__(self, 
                n_layers,
                n_filters_time=25, # L1
                filter_time_length=10,
                n_filters_spat=25, # L1.5
                in_chans=14, # 電極数(L1.5)
                n_filters_2=50, # L2
                filter_length_2=10,
                n_filters_3=100, # L3
                filter_length_3=10,
                n_filters_4=200, # L4
                filter_length_4=10,
                pool_time_length=3, # Pool L1~4 
                pool_time_stride=3, # Pool L1~4
                first_nonlin=nn.ELU, # L1,1.5
                later_nonlin=nn.ELU, # L2~
                drop_prob=0.5,
                batch_norm=True,
                batch_norm_alpha=0.1,
                size_check=False,
                is_single=False,               
                ) :
        '''
        *Deep4Netに追加した引数のみ記載
        n_layers: 層数(1, 1.5, 2, 3, 4)
        size_check: forward時、中間出力のサイズを確認
        is_single: Encoderのみで使用する場合(Decoder用のpool_indices, pool_sizeを返さない)
        '''
        super().__init__() 
        layers = {} 
        conv_stride = 1
        pool_stride = pool_time_stride
        first_pool_class = nn.MaxPool2d
        later_pool_class = nn.MaxPool2d
        
        # L1
        layers['dimshuffle'] = Expression(transpose_time_to_spat) # 14 x 1280 x 1 -> 1 x 1280 x 14
        layers['conv_time'] = nn.Conv2d(in_channels=1, out_channels=n_filters_time, 
                                        kernel_size=(filter_time_length, 1), stride=1)
        if n_layers > 1 :
            # L1.5
            layers['conv_spat'] = nn.Conv2d(n_filters_time, n_filters_spat, (1,in_chans), 
                                            stride=(conv_stride,1),bias= not batch_norm) 
            n_filters_prev = n_filters_spat
        else :
            n_filters_prev = n_filters_time

        if batch_norm:
            layers['bnorm'] = nn.BatchNorm2d(n_filters_prev, momentum=batch_norm_alpha, affine=True,eps=1e-5) 
        layers['conv_nonlin'] = first_nonlin() # Activate
        layers['pool'] = first_pool_class(kernel_size=(pool_time_length,1),stride=(pool_stride,1), return_indices=True) # nn.MaxPool2d

        # L2,3,4
        if n_layers >= 2 :
            n_filters = [n_filters_2, n_filters_3, n_filters_4]
            filter_lengthes = [filter_length_2, filter_length_3, filter_length_4]
            for n_l in range(2, n_layers+1) :
                layers['drop_{}'.format(n_l)] = nn.Dropout(p=drop_prob) # 通すと出力のどれかが0に
                layers['conv_{}'.format(n_l)] = nn.Conv2d(n_filters_prev, n_filters[n_l-2], (filter_lengthes[n_l-2], 1),
                                                          stride=(conv_stride, 1), bias=not batch_norm) 
                if batch_norm:
                    layers['bnorm_{}'.format(n_l)] = nn.BatchNorm2d(n_filters[n_l-2],momentum=batch_norm_alpha,
                                                                    affine=True,eps=1e-5) 
                layers['nonlin_{}'.format(n_l)] = later_nonlin() # Activate
                layers['pool_{}'.format(n_l)] = later_pool_class(kernel_size=(pool_time_length, 1),
                                                                 stride=(pool_stride, 1), return_indices=True) 
                n_filters_prev = n_filters[n_l-2]

        self.layers = nn.ModuleDict(layers)
        self.size_check = size_check
        self.is_single = is_single
    
    # x: Batch x Ch x Value x 1
    def forward(self, x) :
        # Encoder単体
        if self.is_single :
            for name,layer in self.layers.items() :
                if 'pool' in name :
                    x, indices = layer(x)
                else :
                    x = layer(x)
                if self.size_check :
                    print('{}:{}'.format(name, x.size()))
            return x
        # AEの一部
        else :
            pool_indices = [] # Unpool用
            pool_size = [] # Unpool用
            for name,layer in self.layers.items() :
                if 'pool' in name :
                    pool_size.append(x.size())
                    x, indices = layer(x)
                    pool_indices.append(indices)
                else :
                    x = layer(x)
                if self.size_check :
                    print('{}:{}'.format(name, x.size()))
            return x, pool_indices, pool_size
    
    def single(self) :
        '''
        Encoder単体で使用
        '''
        self.is_single = True
#########################################################################################################
# Decoder
class Deep4Decoder(nn.Module) :
    '''
    Deep4Decoder
    ・Deep4Encoderの逆流
    ・ただしdropoutやbnromの場所を調整
    ・引数はEncoderと同じ
    '''
    def __init__(self, 
                n_layers,
                n_filters_time=25, # L1
                filter_time_length=10,
                n_filters_spat=25, # L1.5
                in_chans=14, # 電極数(L1.5)
                n_filters_2=50, # L2
                filter_length_2=10,
                n_filters_3=100, # L3
                filter_length_3=10,
                n_filters_4=200, # L4
                filter_length_4=10,
                pool_time_length=3, # Pool L1~4 
                pool_time_stride=3, # Pool L1~4
                first_nonlin=nn.ELU, # L1,1.5
                later_nonlin=nn.ELU, # L2~
                drop_prob=0.5,
                batch_norm=True,
                batch_norm_alpha=0.1,
                size_check=False,               
                ) :
        # Decoder: 
        super().__init__()
        layers = {}
        conv_stride = 1
        pool_stride = pool_time_stride
        first_pool_class = nn.MaxUnpool2d
        later_pool_class = nn.MaxUnpool2d
        
        # L4,3,2
        # *in,outがencoderの逆に
        # *インデックスが逆順
        if n_layers >= 2 :
            n_filters = [n_filters_spat, n_filters_2, n_filters_3, n_filters_4]
            filter_lengthes = [filter_length_2, filter_length_3, filter_length_4]
            for n_l in range(n_layers, 1, -1) :
                layers['unpool_{}'.format(n_l)] = later_pool_class(kernel_size=(pool_time_length, 1),stride=(pool_stride, 1))
                layers['drop_{}'.format(n_l)] = nn.Dropout(p=drop_prob)
                layers['deconv_{}'.format(n_l)] = nn.ConvTranspose2d(n_filters[n_l-1], n_filters[n_l-2], (filter_lengthes[n_l-2], 1),
                                                                        stride=(conv_stride, 1), bias=not batch_norm) 
                if batch_norm:
                    layers['bnorm_{}'.format(n_l)] = nn.BatchNorm2d(n_filters[n_l-2], momentum=batch_norm_alpha,
                                                                    affine=True,eps=1e-5) 
                layers['nonlin_{}'] = later_nonlin()
        
        # Deconv Pool Block1
        layers['unpool'] = first_pool_class(kernel_size=(pool_time_length,1),stride=(pool_stride,1)) 
        layers['deconv_nonlin'] = first_nonlin ()
        n_filters_conv = n_filters_spat if n_layers > 1 else n_filters_time
        if batch_norm:
            layers['bnorm'] = nn.BatchNorm2d(n_filters_conv,momentum=batch_norm_alpha,affine=True,eps=1e-5)
        if n_layers > 1 :
            layers['deconv_spat'] = nn.ConvTranspose2d(n_filters_spat, n_filters_time, (1,in_chans), 
                                                       stride=(conv_stride,1),bias= not batch_norm)
        layers['deconv_time'] = nn.ConvTranspose2d(in_channels=n_filters_time, out_channels=1, 
                                                   kernel_size=(filter_time_length, 1), stride=1)
        layers['dimshuffle'] = Expression(transpose_time_to_spat) # 1,3次元を入れ替えるだけなので同じで良し

        self.layers = nn.ModuleDict(layers)
        self.size_check = size_check
        
    def forward(self, x, pool_indices, pool_size) :
        p_i = -1 # pool_indices, sizeのカウント(逆順)
        for name,layer in self.layers.items() :
            if 'unpool' in name :
                x = layer(x, indices=pool_indices[p_i], output_size=pool_size[p_i])
                p_i -= 1 
            else :
                x = layer(x)
            if self.size_check :
                print('{}:{}'.format(name, x.size()))
        return x
#########################################################################################################

# Autoencoder
class Deep4AutoEncoder(nn.Module) :
    '''
    Deep4AutoEncoder
    ・Deep4Encoderを改変し、AutoEncoder
    ・Deep4Netから分類機能を削除
    ・その他、機能を一部簡略化
    '''
    def __init__(self, n_layers, **kwargs) :
        '''
        n_layers: 層数
        他引数はEncoder、Decoderで同じ、書くのが面倒なので、**kwargs
        '''
        super().__init__()
        if n_layers > 4 :
            raise ValueError('n_layers = 1, 1.5, 2, 3, 4 ')
        self.encoder = Deep4Encoder(n_layers=n_layers, **kwargs)
        self.decoder = Deep4Decoder(n_layers=n_layers, **kwargs)
        self.size_check = kwargs['size_check'] if 'size_check' in list(kwargs.keys()) else False
    
    def forward(self, x) :
        '''
        x: n_sample x n_ch x n_value x 1
        '''
        if self.size_check :
            print('Encoder')
        x, pool_indices, pool_size = self.encoder(x)
        if self.size_check :
            print('\nDecoder')
        x = self.decoder(x, pool_indices, pool_size)
        return x

      

# Test(from .modules_d4n...をfrom modules_d4n…にしないとエラー)
import torch
if __name__=='__main__': 
    torch.manual_seed(0)
    in_chans = 14
    input = torch.rand((100, in_chans, 1280, 1))
    model = Deep4AutoEncoder(n_layers=2, size_check=True)
    output = model(input)
    print(output[0,:,0,0])
