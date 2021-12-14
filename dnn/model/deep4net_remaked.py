# Pytorchのnn.Sequential形式をnn.Module形式にした
##############################
# BraindecodeDeep4Netのソースでインポートされたもの
import numpy as np
import torch
from torch import nn
from torch.nn import init
import modules_d4n_remaked as bd 
'''
from braindecode.models.modules import Expression, AvgPool2dWithConv, Ensure4d
from braindecode.models.functions import identity, transpose_time_to_spat, squeeze_final_output
from braindecode.util import np_to_var
'''

# self.add_moduleは全部self.moduleに変換
# 一部省略
# 初期パラメータ固定？が機能してなかったため、修正
class Deep4Net(nn.Module) : 
    def __init__(self,
        in_chans, # 電極数
        n_classes, # クラス数
        input_window_samples, # This will determine how many crops are processed in parallel. in Feis
        final_conv_length, # クラス分類: 一回実行して、出力が2次元になる値(errorのper channel: (N x 1)のN)
                           # ただし上記の方法が最適かは不明
        n_filters_time=25, # Fig1 Conv Pool Block1
        n_filters_spat=25,
        filter_time_length=10,
        pool_time_length=3,
        pool_time_stride=3,
        n_filters_2=50, # Fig1 Conv Pool Block2
        filter_length_2=10,
        n_filters_3=100, # Fig1 Conv Pool Block3
        filter_length_3=10,
        n_filters_4=200, # Fig1 Conv Pool Block4
        filter_length_4=10,
        first_nonlin=nn.ELU, # なぜかtorch.F.eluだった、nn.ELUに変更
        first_pool_mode="max",
        first_pool_nonlin=nn.Identity, # なぜかわざわざ自作だった、nn.Identityに変更->first_pool_nonlin()
        later_nonlin=nn.ELU,
        later_pool_mode="max",
        later_pool_nonlin=nn.Identity,
        drop_prob=0.5, # ドロップアウト確率
        # double_time_convs=False, 未使用
        split_first_layer=True,
        batch_norm=True,
        batch_norm_alpha=0.1,
        stride_before_pool=False, 
        init_seed=None # 初期化シード(追加)
        ) :
        super().__init__()

        if final_conv_length == "auto":
            assert input_window_samples is not None 
        
        # ここで本来全引数に対して、self.arg＝arg
        layers = {} # 管理の簡略化を
        if stride_before_pool:
            conv_stride = pool_time_stride
            pool_stride = 1
        else: # Default
            conv_stride = 1
            pool_stride = pool_time_stride

        layers['ensuredims'] = bd.Ensure4d() # Layer1
        pool_class_dict = dict(max=nn.MaxPool2d, mean=bd.AvgPool2dWithConv)
        first_pool_class = pool_class_dict[first_pool_mode]
        later_pool_class = pool_class_dict[later_pool_mode]

        # Change
        # Fig1 Conv Pool Block1
        if split_first_layer: # Default
            layers['dimshuffle'] = bd.Expression(bd.transpose_time_to_spat) # Layer2.1.1
            layers['conv_time'] = nn.Conv2d(in_channels=1, out_channels=n_filters_time, 
                                            kernel_size=(filter_time_length, 1), stride=1) # Layer2.1.2
            layers['conv_spat'] = nn.Conv2d(n_filters_time, n_filters_spat, (1,in_chans), 
                                       stride=(conv_stride,1),bias= not batch_norm) # Layer2.1.3
            n_filters_conv = n_filters_spat
        else:
            layers['conv_time'] = nn.Conv2d(in_chans, n_filters_time, (filter_time_length,1), 
                                       stride=(conv_stride,1),bias=not batch_norm) # Layer2.2.1
            n_filters_conv = n_filters_time
        
        if batch_norm: # Default
            layers['bnorm'] = nn.BatchNorm2d(n_filters_conv,momentum=batch_norm_alpha,affine=True,eps=1e-5) # Layer3.0
        layers['conv_nonlin'] = first_nonlin() # Layer3.1
        layers['pool'] = first_pool_class(kernel_size=(pool_time_length,1),stride=(pool_stride,1)) # Layer3.2
        layers['pool_nonlin'] = first_pool_nonlin() # Layer3.3

        # # Fig1 Conv Pool Block2,3,4
        # add_conv_pool_blockを分解(suffixができない)
        n_filters = [n_filters_2, n_filters_3, n_filters_4]
        n_filters_before = n_filters_conv
        filter_lengthes = [filter_length_2, filter_length_3, filter_length_4]
        for i in range(len(n_filters)) :
            layers['drop_{:d}'.format(i+2)] = nn.Dropout(p=drop_prob) # Layer4.[1-3].1
            layers['conv_{:d}'.format(i+2)] = nn.Conv2d(n_filters_before, n_filters[i],(filter_lengthes[i], 1),
                                                      stride=(conv_stride, 1), bias=not batch_norm) # Layer4.[1-3].2
            if batch_norm:
                layers['bnorm_{:d}'.format(i+2)] = nn.BatchNorm2d(n_filters[i],momentum=batch_norm_alpha,
                                                                affine=True,eps=1e-5) # Layer4.[1-3].(2.5)
            layers['nonlin_{:d}'.format(i+2)] = later_nonlin() # Layer4.[1-3].3
            layers['pool_{:d}'.format(i+2)] = later_pool_class(kernel_size=(pool_time_length, 1),stride=(pool_stride, 1)) # Layer4.[1-3].4
            layers['pool_nonlin_{:d}'.format(i+2)] = later_pool_nonlin() # Layer4.[1-3].5
            n_filters_before = n_filters[i] 

        # self.add_module('drop_classifier', nn.Dropout(p=self.drop_prob)
        # Fig1 Classification Layer
        if final_conv_length == "auto": # not Default
            self.eval()
            out = self(bd.np_to_th(np.ones((1, in_chans, input_window_samples, 1),dtype=np.float32)))
            n_out_time = out.cpu().data.numpy().shape[2]
            final_conv_length = n_out_time
        layers['conv_classifier'] = nn.Conv2d(n_filters[-1],n_classes,(final_conv_length, 1),bias=True) # Layer5.1
        layers['softmax'] = nn.LogSoftmax(dim=1) # Layer5.2
        layers['squeeze'] = bd.Expression(bd.squeeze_final_output) # Layer5.3
        
        self.layers = nn.ModuleDict(layers)

        
        if init_seed is not None :
            torch.random.manual_seed(init_seed) # <-これを追加したら初期パラメータ固定出来た
            # 重みの初期化、ただし上記の追加しないと機能していなかった
            # Initialization, xavier is same as in our paper...
            # was default from lasagne
            
            # Xavier: torchの初期化を司る何か
            # lasagneは、Theanoのニューラルネットワークを構築し、訓練するための軽量ライブラリ
            
            # xavier_uniform:
            # Fills the input Tensor with values drawn from the uniform distribution u(a,b)
            # xavier_constant_:
            # Fills the input Tensor with the value val.
            init.xavier_uniform_(self.layers['conv_time'].weight, gain=1) 
            # maybe no bias in case of no split layer and batch norm
            if split_first_layer or (not batch_norm):
                init.constant_(self.layers['conv_time'].bias, 0) 
            if split_first_layer:
                init.xavier_uniform_(self.layers['conv_spat'].weight, gain=1)
                if not batch_norm:
                    init.constant_(self.layers['conv_spat.bias'], 0)
            if batch_norm:
                init.constant_(self.layers['bnorm'].weight, 1)
                init.constant_(self.layers['bnorm'].bias, 0)
            
            param_dict = dict(list(self.named_parameters()))
            for block_nr in range(2, 5):
                conv_weight = param_dict["layers.conv_{:d}.weight".format(block_nr)]
                init.xavier_uniform_(conv_weight, gain=1)
                if not batch_norm:
                    conv_bias = param_dict["layers.conv_{:d}.bias".format(block_nr)]
                    init.constant_(conv_bias, 0)
                else:
                    bnorm_weight = param_dict["layers.bnorm_{:d}.weight".format(block_nr)]
                    bnorm_bias = param_dict["layers.bnorm_{:d}.bias".format(block_nr)]
                    init.constant_(bnorm_weight, 1)
                    init.constant_(bnorm_bias, 0)

            init.xavier_uniform_(self.layers['conv_classifier'].weight, gain=1)
            init.constant_(self.layers['conv_classifier'].bias, 0)

    # x: Batch x Ch x Value x 1
    def forward(self, x, size_check=False) :
        for name,layer in self.layers.items() :
            x = layer(x)
            if size_check :
                print('{}:{}'.format(name, x.size()))
        return x



# Deep4AutoEncoder
# Encoder
class Deep4Encoder(nn.Module) :
    '''
    Deep4Encoder
    ・Deep4Netデフォルトと流れは同じ
    ・分類器削除
    ・デフォルトでない分岐を削除、使わない引数削除
    '''
    def __init__(self, 
                in_chans, # 電極数
                n_filters_time=25, # Conv Pool Block1
                n_filters_spat=25,
                filter_time_length=10,
                pool_time_length=3,
                pool_time_stride=3,
                n_filters_2=50, # Conv Pool Block2
                filter_length_2=10,
                n_filters_3=100, # Conv Pool Block3
                filter_length_3=10,
                n_filters_4=200, # Conv Pool Blofck4
                filter_length_4=10,
                first_nonlin=nn.ELU,
                # first_pool_mode="max",
                # first_pool_nonlin=nn.Identity, 
                later_nonlin=nn.ELU,
                # later_pool_mode="max",
                # later_pool_nonlin=nn.Identity,
                drop_prob=0.5,
                batch_norm=True,
                batch_norm_alpha=0.1,
                size_check=False
                ) :
        # Encoder: 
        # block1: in->dim->conv->conv->bnorm->elu->maxpool->
        # block2,3,4: (drop->conv->bnorm->elu->maxpool->)x3 ->out

        super().__init__() 
        layers = {} 
        conv_stride = 1
        pool_stride = pool_time_stride
        # layers['ensuredims'] = bd.Ensure4d() いらなそう
        # pool_class_dict = dict(max=nn.MaxPool2d, mean=bd.AvgPool2dWithConv) オリジナル層 AvgPool2dWithConvのDeconv層が作れないから
        first_pool_class = nn.MaxPool2d# pool_class_dict[first_pool_mode]
        later_pool_class = nn.MaxPool2d# pool_class_dict[later_pool_mode]
        
        # Conv Pool Block1
        layers['dimshuffle'] = bd.Expression(bd.transpose_time_to_spat)
        layers['conv_time'] = nn.Conv2d(in_channels=1, out_channels=n_filters_time, 
                                        kernel_size=(filter_time_length, 1), stride=1)
        layers['conv_spat'] = nn.Conv2d(n_filters_time, n_filters_spat, (1,in_chans), 
                                        stride=(conv_stride,1),bias= not batch_norm) 
        n_filters_conv = n_filters_spat
        if batch_norm:
            layers['bnorm'] = nn.BatchNorm2d(n_filters_conv,momentum=batch_norm_alpha,affine=True,eps=1e-5) 
        layers['conv_nonlin'] = first_nonlin() #  nn.ELU
        layers['pool'] = first_pool_class(kernel_size=(pool_time_length,1),stride=(pool_stride,1), return_indices=True) # nn.MaxPool2d
        # layers['pool_nonlin'] = first_pool_nonlin # nn.Identity(デフォルト)は無意味のため

        # Conv Pool Block2,3,4
        n_filters = [n_filters_2, n_filters_3, n_filters_4]
        n_filters_before = n_filters_conv
        filter_lengthes = [filter_length_2, filter_length_3, filter_length_4]
        for i in range(2,5) :
            layers['drop_{:d}'.format(i)] = nn.Dropout(p=drop_prob) # 通すと出力のどれかが0に
            layers['conv_{:d}'.format(i)] = nn.Conv2d(n_filters_before, n_filters[i-2],(filter_lengthes[i-2], 1),
                                                        stride=(conv_stride, 1), bias=not batch_norm) 
            if batch_norm:
                layers['bnorm_{:d}'.format(i)] = nn.BatchNorm2d(n_filters[i-2],momentum=batch_norm_alpha,
                                                                  affine=True,eps=1e-5) 
            layers['nonlin_{:d}'.format(i)] = later_nonlin() # nn.ELU
            layers['pool_{:d}'.format(i)] = later_pool_class(kernel_size=(pool_time_length, 1),stride=(pool_stride, 1), return_indices=True) # nn.MaxPool2d
            # layers['pool_nonlin_{:d}'.format(i+2)] = later_pool_nonlin # nn.Identity
            n_filters_before = n_filters[i-2] 

        self.layers = nn.ModuleDict(layers)
        self.size_check = size_check
    
    # x: Batch x Ch x Value x 1
    def forward(self, x) :
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

# Decoder
class Deep4Decoder(nn.Module) :
    '''
    Deep4Decoder
    ・Deep4Encoderの逆流
    ・ただしdropoutやbnromの場所を調整
    ・引数はEncoderと同じ
    '''
    def __init__(self, 
                in_chans, # 電極数
                n_filters_time=25, # DeConv Pool Block1
                n_filters_spat=25,
                filter_time_length=10,
                pool_time_length=3,
                pool_time_stride=3,
                n_filters_2=50, # DeConv Pool Block2
                filter_length_2=10,
                n_filters_3=100, # DeConv Pool Block3
                filter_length_3=10,
                n_filters_4=200, # DeConv Pool Block4
                filter_length_4=10,
                first_nonlin=nn.ELU,
                later_nonlin=nn.ELU,
                drop_prob=0.5,
                batch_norm=True,
                batch_norm_alpha=0.1,
                size_check=False
                ) :
        # Decoder: 
        # block4,3,2: in-> (maxunpool->drop->deconv->bnrom->elu->)x3->
        # block1: ->maxunpool->deconv->bnorm->elu->deconv->dim->out
        # *Encoderの形からなんとなく、dropやbnormの場所は適切なのか？

        super().__init__()
        layers = {}
        conv_stride = 1
        pool_stride = pool_time_stride
        first_pool_class = nn.MaxUnpool2d
        later_pool_class = nn.MaxUnpool2d
        n_filters_conv = n_filters_spat

        # Deconv Pool Block4,3,2
        # *in,outがencoderの逆に
        # *インデックスが逆順
        n_filters = [n_filters_conv, n_filters_2, n_filters_3] # in,outの関係で変更
        n_filters_before =  n_filters_4 # 
        filter_lengthes = [filter_length_2, filter_length_3, filter_length_4]
        for i in range(4, 1, -1) : # 4, 3, 2
            layers['unpool_{:d}'.format(i)] = later_pool_class(kernel_size=(pool_time_length, 1),stride=(pool_stride, 1))
            layers['drop_{:d}'.format(i)] = nn.Dropout(p=drop_prob)
            layers['deconv_{:d}'.format(i)] = nn.ConvTranspose2d(n_filters_before, n_filters[i-2],(filter_lengthes[i-2], 1),
                                                                 stride=(conv_stride, 1), bias=not batch_norm) 
            if batch_norm:
                layers['bnorm_{:d}'.format(i)] = nn.BatchNorm2d(n_filters[i-2],momentum=batch_norm_alpha,
                                                                affine=True,eps=1e-5) 
            layers['nonlin_{:d}'.format(i)] = later_nonlin()
            n_filters_before = n_filters[i-2]
        
        # Deconv Pool Block1
        layers['unpool'] = first_pool_class(kernel_size=(pool_time_length,1),stride=(pool_stride,1)) 
        layers['conv_nonlin'] = first_nonlin ()
        if batch_norm:
            layers['bnorm'] = nn.BatchNorm2d(n_filters_conv,momentum=batch_norm_alpha,affine=True,eps=1e-5)
        layers['deconv_spat'] = nn.ConvTranspose2d(n_filters_spat, n_filters_time, (1,in_chans), 
                                                   stride=(conv_stride,1),bias= not batch_norm)
        layers['deconv_time'] = nn.ConvTranspose2d(in_channels=n_filters_time, out_channels=1, 
                                                   kernel_size=(filter_time_length, 1), stride=1)
        layers['dimshuffle'] = bd.Expression(bd.transpose_time_to_spat) # 1,3次元を入れ替えるだけなので同じで良し

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


# Autoencoder
class Deep4AutoEncoder(nn.Module) :
    '''
    Deep4AutoEncoder
    ・Deep4Encoderを改変し、AutoEncoder
    ・Deep4Netから分類機能を削除
    ・その他、機能を一部簡略化
    '''
    def __init__(self, in_chans, **kwargs) :
        '''
        in_chans: 電極数(必須)
        その他引数はEncoder、Decoderで同じ、書くのが面倒なので、**kwargs
        '''
        super().__init__()
        self.encoder = Deep4Encoder(in_chans=in_chans, **kwargs)
        self.decoder = Deep4Decoder(in_chans=in_chans, **kwargs)
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

# Test
if __name__=='__main__': 
    in_chans = 14
    input = torch.rand((100, in_chans, 256, 1))
    model = Deep4AutoEncoder(in_chans=in_chans, size_check=True)
    output = model(input)
    # 入力サイズtimeに強く依存
    # deconv3でエラー        
