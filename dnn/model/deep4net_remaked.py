# Deep4Net, Deep4AutoEncoder
# Deep4Net: 元のDeep4Netのnn.Sequential形式をnn.Module形式にした
# Deep4AutoEncoder: Deep4NetをAutoEncoderに
##############################
import numpy as np
import torch
from torch import nn
from torch.nn import init
# import modules_d4n_remaked as bd # BraindecodeDeep4Netで使われてたクラス、関数をまとめた
from .modules_d4n_remaked import Ensure4d, Expression, AvgPool2dWithConv, \
                                 transpose_time_to_spat, squeeze_final_output, np_to_th

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

        layers['ensuredims'] = Ensure4d() # Layer1
        pool_class_dict = dict(max=nn.MaxPool2d, mean=AvgPool2dWithConv)
        first_pool_class = pool_class_dict[first_pool_mode]
        later_pool_class = pool_class_dict[later_pool_mode]

        # Change
        # Fig1 Conv Pool Block1
        if split_first_layer: # Default
            layers['dimshuffle'] = Expression(transpose_time_to_spat) # Layer2.1.1
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
            out = self(np_to_th(np.ones((1, in_chans, input_window_samples, 1),dtype=np.float32)))
            n_out_time = out.cpu().data.numpy().shape[2]
            final_conv_length = n_out_time
        layers['conv_classifier'] = nn.Conv2d(n_filters[-1],n_classes,(final_conv_length, 1),bias=True) # Layer5.1
        layers['softmax'] = nn.LogSoftmax(dim=1) # Layer5.2
        layers['squeeze'] = Expression(squeeze_final_output) # Layer5.3
        
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



