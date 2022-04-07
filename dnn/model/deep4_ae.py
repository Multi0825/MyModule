# Deep4NetをAutoEncoderに改良(修士中間で使用した物とほぼ同一,ただしbatch_norm2の位置が前後)
# 畳み込み層数を1,2,3,4,5で変更可能
# ＊引数の名前がDeep4Netとも修士で使用したものとも違う

from multiprocessing.sharedctypes import Value
import torch.nn as nn
# import modules_d4n_remaked as bd # BraindecodeDeep4Netで使われてたクラス、関数をまとめた
from .modules_d4n_remaked import Ensure4d, Expression, AvgPool2dWithConv, \
                                 transpose_time_to_spat, squeeze_final_output, np_to_th


#########################################################################################################
# Encoder
class Deep4Encoder(nn.Module) :
    '''
    Deep4AutoEncoderのEncoder
    単体で使用する場合はsingleを呼び出す(unpooling用の変数を返さない)
    '''
    def __init__(self, 
                n_convs,
                n_filters_1=25, # Conv1
                filter_length_1=10,
                in_chans=14, # 電極数(filter_length_2)
                n_filters_2=25, # Conv2
                n_filters_3=50, # Conv3
                filter_length_3=10,
                n_filters_4=100, # Conv4
                filter_length_4=10,
                n_filters_5=200, # Conv5
                filter_length_5=10,
                pool_time_length=3, # Pool2~ 
                pool_time_stride=3, # Pool2~
                first_nonlin=nn.ELU, # 2
                later_nonlin=nn.ELU, # 3~
                drop_prob=0.5,
                batch_norm=True,
                batch_norm_alpha=0.1,
                size_check=False,
                is_single=False,               
                ) :
        super().__init__() 
        layers = {} 
        conv_stride = 1
        first_pool_class = nn.MaxPool2d
        later_pool_class = nn.MaxPool2d
        pool_kernel_size = (pool_time_length,1)
        pool_stride = (pool_time_stride,1)
        n_filters = [n_filters_1, n_filters_2, n_filters_3, n_filters_4, n_filters_5]
        kernel_sizes = [(filter_length_1, 1), (1, in_chans), 
                        (filter_length_3, 1), (filter_length_4, 1), (filter_length_5, 1)]

        layers['dimshuffle'] = Expression(transpose_time_to_spat) # 14 x 1280 x 1 -> 1 x 1280 x 14
        
        # Conv1
        layers['conv_1'] = nn.Conv2d(in_channels=1, out_channels=n_filters[0], 
                                     kernel_size=kernel_sizes[0], stride=1)
        # Conv2
        if n_convs > 1 :
            layers['conv_2'] = nn.Conv2d(n_filters[0], n_filters[1], kernel_sizes[1], 
                                            stride=(conv_stride,1), bias= not batch_norm) 
        n_filters_prev = n_filters[1] if n_convs>1 else n_filters[0]
        if batch_norm:
            layers['bnorm_2'] = nn.BatchNorm2d(n_filters_prev, momentum=batch_norm_alpha, affine=True,eps=1e-5) 
        layers['nonlin_2'] = first_nonlin() # Activate
        layers['pool_2'] = first_pool_class(kernel_size=pool_kernel_size, stride=pool_stride, return_indices=True) # nn.MaxPool2d

        # Conv3~5
        for n_c in range(3, n_convs+1) :
            layers['drop_{}'.format(n_c)] = nn.Dropout(p=drop_prob) # 通すと出力のどれかが0に
            layers['conv_{}'.format(n_c)] = nn.Conv2d(n_filters[n_c-2], n_filters[n_c-1], kernel_sizes[n_c-1],
                                                      stride=(conv_stride, 1), bias=not batch_norm) 
            if batch_norm:
                layers['bnorm_{}'.format(n_c)] = nn.BatchNorm2d(n_filters[n_c-1],momentum=batch_norm_alpha,
                                                                affine=True,eps=1e-5) 
            layers['nonlin_{}'.format(n_c)] = later_nonlin() # Activate
            layers['pool_{}'.format(n_c)] = later_pool_class(kernel_size=pool_kernel_size,
                                                             stride=pool_stride, return_indices=True) 

        self.layers = nn.ModuleDict(layers)
        self.size_check = size_check
        self.is_single = is_single
    
    # x: Batch x Ch x Value x 1
    def forward(self, x) :
        # Encoder単体として
        if self.is_single :
            for name,layer in self.layers.items() :
                if 'pool' in name :
                    x, indices = layer(x)
                else :
                    x = layer(x)
                if self.size_check :
                    print('{}:{}'.format(name, x.size()))
            return x
        # AEの一部として
        else :
            pool_indices = [] # Unpooling用
            pool_sizes = [] # Unpooling用
            for name,layer in self.layers.items() :
                if 'pool' in name :
                    pool_sizes.append(x.size())
                    x, indices = layer(x)
                    pool_indices.append(indices)
                else :
                    x = layer(x)
                if self.size_check :
                    print('{}:{}'.format(name, x.size()))
            return x, pool_indices, pool_sizes
    
    def single(self) :
        '''
        Encoder単体で使用するときに呼び出す
        '''
        self.is_single = True

#########################################################################################################
# Decoder
class Deep4Decoder(nn.Module) :
    '''
    Deep4AutoEncoderのDecoder(単体使用不可)
    '''
    def __init__(self, 
                n_convs,
                n_filters_1=25, # Deconv1
                filter_length_1=10,
                in_chans=14, # 電極数(filter_length_2)
                n_filters_2=25, # Deconv2
                n_filters_3=50, # Deconv3
                filter_length_3=10,
                n_filters_4=100, # Deconv4
                filter_length_4=10,
                n_filters_5=200, # Deconv5
                filter_length_5=10,
                pool_time_length=3, # Unpool2~ 
                pool_time_stride=3, # Unpool2~
                first_nonlin=nn.ELU, # 2
                later_nonlin=nn.ELU, # 3~
                drop_prob=0.5,
                batch_norm=True,
                batch_norm_alpha=0.1,
                size_check=False,            
                ) : 
        super().__init__()
        layers = {}
        conv_stride = 1
        pool_stride = pool_time_stride
        first_pool_class = nn.MaxUnpool2d
        later_pool_class = nn.MaxUnpool2d
        pool_kernel_size = (pool_time_length,1)
        pool_stride = (pool_time_stride,1)
        n_filters = [n_filters_1, n_filters_2, n_filters_3, n_filters_4, n_filters_5]
        kernel_sizes = [(filter_length_1, 1), (1, in_chans), 
                        (filter_length_3, 1), (filter_length_4, 1), (filter_length_5, 1)]
        
        # Deconv5~3(*in,outがencoderの逆)
        for n_c in range(n_convs, 2, -1) : 
            layers['unpool_{}'.format(n_c)] = later_pool_class(kernel_size=pool_kernel_size,stride=pool_stride)
            layers['drop_{}'.format(n_c)] = nn.Dropout(p=drop_prob)
            layers['deconv_{}'.format(n_c)] = nn.ConvTranspose2d(n_filters[n_c-1], n_filters[n_c-2], kernel_sizes[n_c-1],
                                                                 stride=(conv_stride, 1), bias=not batch_norm) 
            if batch_norm:
                layers['bnorm_{}'.format(n_c)] = nn.BatchNorm2d(n_filters[n_c-2], momentum=batch_norm_alpha,
                                                                affine=True,eps=1e-5) 
            layers['nonlin_{}'.format(n_c)] = later_nonlin()
        
        # Deconv2, 1
        layers['unpool_2'] = first_pool_class(kernel_size=pool_kernel_size,stride=pool_stride) 
        if n_convs > 1 :
            layers['deconv_2'] = nn.ConvTranspose2d(n_filters[1], n_filters[0], kernel_sizes[1], 
                                                    stride=(conv_stride,1),bias= not batch_norm)
        n_filters_prev = n_filters[1] if n_convs > 1 else n_filters[0] 
        if batch_norm:
            layers['bnorm_2'] = nn.BatchNorm2d(n_filters_prev, momentum=batch_norm_alpha,affine=True,eps=1e-5)
        layers['deconv_nonlin_2'] = first_nonlin ()
        
        layers['deconv_1'] = nn.ConvTranspose2d(in_channels=n_filters[0], out_channels=1, 
                                                   kernel_size=kernel_sizes[0], stride=1)
        layers['dimshuffle'] = Expression(transpose_time_to_spat) # 1,3次元を入れ替えるだけなので同じで良し

        self.layers = nn.ModuleDict(layers)
        self.size_check = size_check
        
    def forward(self, x, pool_indices, pool_sizes) :
        p_i = -1 # pool_indices, sizeのカウント(逆順)
        for name,layer in self.layers.items() :
            if 'unpool' in name :
                x = layer(x, indices=pool_indices[p_i], output_size=pool_sizes[p_i])
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
    ・Deep4Netを改変し、AutoEncoder化
    ・Deep4Netから分類機能を削除、機能を一部簡略化
    '''
    def __init__(self, n_convs, **kwargs) :
        '''
        引数はEncoder,Decoder共通(*Deep4Netから追加、変更した引数のみ記載)
        n_convs: 畳み込み層数(1, 2, 3, 4, 5)
        n_filters_N: 各畳み込み層の出力チャンネル数
        filter_length_N: 各畳み込み層の出力のカーネルサイズに関連(2のみ電極方向)
        size_check: forward時、中間出力のサイズを確認
        is_single: Encoderのみで使用する場合(Decoder用のpool_indices, pool_sizeを返さない)
        *以下略
        '''
        super().__init__()
        if (n_convs<1) or (n_convs>5) :
            raise ValueError('n_convs = 1, 2, 3, 4, 5')
        self.encoder = Deep4Encoder(n_convs=n_convs, **kwargs)
        self.decoder = Deep4Decoder(n_convs=n_convs, **kwargs)
        self.size_check = kwargs['size_check'] if 'size_check' in list(kwargs.keys()) else False
    
    def forward(self, x) :
        '''
        x: n_samples x n_chs x n_values x 1
        '''
        if self.size_check :
            print('Encoder')
        x, pool_indices, pool_sizes = self.encoder(x)
        if self.size_check :
            print('\nDecoder')
        x = self.decoder(x, pool_indices, pool_sizes)
        return x

      

# Test(from .modules_d4n...をfrom modules_d4n…にしないとエラー)
import torch
if __name__=='__main__': 
    torch.manual_seed(0)
    in_chans = 14
    input = torch.rand((100, in_chans, 1280, 1))
    model = Deep4AutoEncoder(n_convs=2, size_check=True)
    output = model(input)
    print(output[0,:,0,0])
