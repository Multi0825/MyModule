# 修士中間で使用したもの全部乗せる(L1~L4)
# Deep4Net, Deep4AutoEncoder
# Deep4Net: 元のDeep4Netのnn.Sequential形式をnn.Module形式にした
# Deep4AutoEncoder: Deep4NetをAutoEncoderに
##############################
from torch import nn
# import modules_d4n_remaked as bd # BraindecodeDeep4Netで使われてたクラス、関数をまとめた
from .modules_d4n_remaked import Ensure4d, Expression, AvgPool2dWithConv, \
                                 transpose_time_to_spat, squeeze_final_output, np_to_th

################################################# L1 ##################################################
# Deep4AutoEncoder
# Encoder
class Deep4EncoderL1(nn.Module) :
    '''
    Deep4Encoder
    ・Deep4Netデフォルトと流れは同じ
    ・分類器削除
    ・デフォルトでない分岐を削除、使わない引数削除
    when using encoder only, call single() or switch 'is_single' True
    '''
    def __init__(self, 
                n_filters_time=25, # Conv Pool Block1
                filter_time_length=10,
                pool_time_length=3,
                pool_time_stride=3,
                first_nonlin=nn.ELU,
                batch_norm=True,
                batch_norm_alpha=0.1,
                size_check=False,
                is_single=False
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
        layers['dimshuffle'] = Expression(transpose_time_to_spat)
        layers['conv_time'] = nn.Conv2d(in_channels=1, out_channels=n_filters_time, 
                                        kernel_size=(filter_time_length, 1), stride=1) 
        n_filters_conv = n_filters_time
        if batch_norm:
            layers['bnorm'] = nn.BatchNorm2d(n_filters_conv,momentum=batch_norm_alpha,affine=True,eps=1e-5) 
        layers['conv_nonlin'] = first_nonlin() #  nn.ELU
        layers['pool'] = first_pool_class(kernel_size=(pool_time_length,1),stride=(pool_stride,1), return_indices=True) # nn.MaxPool2d
        # layers['pool_nonlin'] = first_pool_nonlin # nn.Identity(デフォルト)は無意味のため

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

# Decoder
class Deep4DecoderL1(nn.Module) :
    '''
    Deep4Decoder
    ・Deep4Encoderの逆流
    ・ただしdropoutやbnromの場所を調整
    ・引数はEncoderと同じ
    '''
    def __init__(self, 
                n_filters_time=25, # DeConv Pool Block1
                filter_time_length=10,
                pool_time_length=3,
                pool_time_stride=3,
                first_nonlin=nn.ELU,
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
        n_filters_conv = n_filters_time

        # Deconv Pool Block1
        layers['unpool'] = first_pool_class(kernel_size=(pool_time_length,1),stride=(pool_stride,1)) 
        layers['deconv_nonlin'] = first_nonlin ()
        if batch_norm:
            layers['bnorm'] = nn.BatchNorm2d(n_filters_conv,momentum=batch_norm_alpha,affine=True,eps=1e-5)
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

################################################# L1.5 ##################################################
# Deep4AutoEncoder
# Encoder
class Deep4EncoderL1_5(nn.Module) :
    '''
    Deep4Encoder
    ・Deep4Netデフォルトと流れは同じ
    ・分類器削除
    ・デフォルトでない分岐を削除、使わない引数削除
    when using encoder only, call single() or switch 'is_single' True
    '''
    def __init__(self, 
                in_chans=14, # 電極数
                n_filters_time=25, # Conv Pool Block1
                n_filters_spat=25,
                filter_time_length=10,
                pool_time_length=3,
                pool_time_stride=3,
                first_nonlin=nn.ELU,
                batch_norm=True,
                batch_norm_alpha=0.1,
                size_check=False,
                is_single=False
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
        layers['dimshuffle'] = Expression(transpose_time_to_spat)
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

# Decoder
class Deep4DecoderL1_5(nn.Module) :
    '''
    Deep4Decoder
    ・Deep4Encoderの逆流
    ・ただしdropoutやbnromの場所を調整
    ・引数はEncoderと同じ
    '''
    def __init__(self, 
                in_chans=14, # 電極数
                n_filters_time=25, # DeConv Pool Block1
                n_filters_spat=25,
                filter_time_length=10,
                pool_time_length=3,
                pool_time_stride=3,
                first_nonlin=nn.ELU,
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

        # Deconv Pool Block1
        layers['unpool'] = first_pool_class(kernel_size=(pool_time_length,1),stride=(pool_stride,1)) 
        layers['deconv_nonlin'] = first_nonlin ()
        if batch_norm:
            layers['bnorm'] = nn.BatchNorm2d(n_filters_conv,momentum=batch_norm_alpha,affine=True,eps=1e-5)
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

################################################# L2 ##################################################
# Deep4AutoEncoder
# Encoder
class Deep4EncoderL2(nn.Module) :
    '''
    Deep4Encoder
    ・Deep4Netデフォルトと流れは同じ
    ・分類器削除
    ・デフォルトでない分岐を削除、使わない引数削除
    when using encoder only, call single() or switch 'is_single' True
    '''
    def __init__(self, 
                in_chans=14, # 電極数
                n_filters_time=25, # Conv Pool Block1
                n_filters_spat=25,
                filter_time_length=10,
                pool_time_length=3,
                pool_time_stride=3,
                n_filters_2=50, # Conv Pool Block2
                filter_length_2=10,
                first_nonlin=nn.ELU,
                later_nonlin=nn.ELU,
                drop_prob=0.5,
                batch_norm=True,
                batch_norm_alpha=0.1,
                size_check=False,
                is_single=False
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
        layers['dimshuffle'] = Expression(transpose_time_to_spat)
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
        layers['drop_2'] = nn.Dropout(p=drop_prob) # 通すと出力のどれかが0に
        layers['conv_2'] = nn.Conv2d(n_filters_conv, n_filters_2,(filter_length_2, 1),
                                                    stride=(conv_stride, 1), bias=not batch_norm) 
        if batch_norm:
            layers['bnorm_2'] = nn.BatchNorm2d(n_filters_2,momentum=batch_norm_alpha,
                                                                affine=True,eps=1e-5) 
        layers['nonlin_2'] = later_nonlin() # nn.ELU
        layers['pool_2'] = later_pool_class(kernel_size=(pool_time_length, 1),stride=(pool_stride, 1), return_indices=True) # nn.MaxPool2d

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

# Decoder
class Deep4DecoderL2(nn.Module) :
    '''
    Deep4Decoder
    ・Deep4Encoderの逆流
    ・ただしdropoutやbnromの場所を調整
    ・引数はEncoderと同じ
    '''
    def __init__(self, 
                in_chans=14, # 電極数
                n_filters_time=25, # DeConv Pool Block1
                n_filters_spat=25,
                filter_time_length=10,
                pool_time_length=3,
                pool_time_stride=3,
                n_filters_2=50, # DeConv Pool Block2
                filter_length_2=10,
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
        layers['unpool_2'] = later_pool_class(kernel_size=(pool_time_length, 1),stride=(pool_stride, 1))
        layers['drop_2'] = nn.Dropout(p=drop_prob)
        layers['deconv_2'] = nn.ConvTranspose2d(n_filters_2, n_filters_conv,(filter_length_2, 1),
                                                                stride=(conv_stride, 1), bias=not batch_norm) 
        if batch_norm:
            layers['bnorm_2'] = nn.BatchNorm2d(n_filters_conv,momentum=batch_norm_alpha,
                                                            affine=True,eps=1e-5) 
        layers['nonlin_2'] = later_nonlin()
        
        # Deconv Pool Block1
        layers['unpool'] = first_pool_class(kernel_size=(pool_time_length,1),stride=(pool_stride,1)) 
        layers['deconv_nonlin'] = first_nonlin ()
        if batch_norm:
            layers['bnorm'] = nn.BatchNorm2d(n_filters_conv,momentum=batch_norm_alpha,affine=True,eps=1e-5)
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

################################################# L3 ##################################################
# Deep4AutoEncoder
# Encoder
class Deep4EncoderL3(nn.Module) :
    '''
    Deep4Encoder
    ・Deep4Netデフォルトと流れは同じ
    ・分類器削除
    ・デフォルトでない分岐を削除、使わない引数削除
    when using encoder only, call single() or switch 'is_single' True
    '''
    def __init__(self, 
                in_chans=14, # 電極数
                n_filters_time=25, # Conv Pool Block1
                n_filters_spat=25,
                filter_time_length=10,
                pool_time_length=3,
                pool_time_stride=3,
                n_filters_2=50, # Conv Pool Block2
                filter_length_2=10,
                n_filters_3=100, # Conv Pool Block3
                filter_length_3=10,
                first_nonlin=nn.ELU,
                # first_pool_mode="max",
                # first_pool_nonlin=nn.Identity, 
                later_nonlin=nn.ELU,
                # later_pool_mode="max",
                # later_pool_nonlin=nn.Identity,
                drop_prob=0.5,
                batch_norm=True,
                batch_norm_alpha=0.1,
                size_check=False,
                is_single=False
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
        layers['dimshuffle'] = Expression(transpose_time_to_spat)
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
        layers['drop_2'] = nn.Dropout(p=drop_prob) # 通すと出力のどれかが0に
        layers['conv_2'] = nn.Conv2d(n_filters_conv, n_filters_2,(filter_length_2, 1),
                                                    stride=(conv_stride, 1), bias=not batch_norm) 
        if batch_norm:
            layers['bnorm_2'] = nn.BatchNorm2d(n_filters_2,momentum=batch_norm_alpha,
                                                                affine=True,eps=1e-5) 
        layers['nonlin_2'] = later_nonlin() # nn.ELU
        layers['pool_2'] = later_pool_class(kernel_size=(pool_time_length, 1),stride=(pool_stride, 1), return_indices=True) # nn.MaxPool2d
        
        layers['drop_3'] = nn.Dropout(p=drop_prob) # 通すと出力のどれかが0に
        layers['conv_3'] = nn.Conv2d(n_filters_2, n_filters_3,(filter_length_3, 1),
                                                    stride=(conv_stride, 1), bias=not batch_norm) 
        if batch_norm:
            layers['bnorm_3'] = nn.BatchNorm2d(n_filters_3,momentum=batch_norm_alpha,
                                                                affine=True,eps=1e-5) 
        layers['nonlin_3'] = later_nonlin() # nn.ELU
        layers['pool_3'] = later_pool_class(kernel_size=(pool_time_length, 1),stride=(pool_stride, 1), return_indices=True) # nn.MaxPool2d

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

# Decoder
class Deep4DecoderL3(nn.Module) :
    '''
    Deep4Decoder
    ・Deep4Encoderの逆流
    ・ただしdropoutやbnromの場所を調整
    ・引数はEncoderと同じ
    '''
    def __init__(self, 
                in_chans=14, # 電極数
                n_filters_time=25, # DeConv Pool Block1
                n_filters_spat=25,
                filter_time_length=10,
                pool_time_length=3,
                pool_time_stride=3,
                n_filters_2=50, # DeConv Pool Block2
                filter_length_2=10,
                n_filters_3=100, # DeConv Pool Block3
                filter_length_3=10,
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
        layers['unpool_3'] = later_pool_class(kernel_size=(pool_time_length, 1),stride=(pool_stride, 1))
        layers['drop_3'] = nn.Dropout(p=drop_prob)
        layers['deconv_3'] = nn.ConvTranspose2d(n_filters_3, n_filters_2,(filter_length_3, 1),
                                                                stride=(conv_stride, 1), bias=not batch_norm) 
        if batch_norm:
            layers['bnorm_3'] = nn.BatchNorm2d(n_filters_2, momentum=batch_norm_alpha,
                                                            affine=True,eps=1e-5) 
        layers['nonlin_3'] = later_nonlin()

        layers['unpool_2'] = later_pool_class(kernel_size=(pool_time_length, 1),stride=(pool_stride, 1))
        layers['drop_2'] = nn.Dropout(p=drop_prob)
        layers['deconv_2'] = nn.ConvTranspose2d(n_filters_2, n_filters_conv,(filter_length_2, 1),
                                                                stride=(conv_stride, 1), bias=not batch_norm) 
        if batch_norm:
            layers['bnorm_2'] = nn.BatchNorm2d(n_filters_conv,momentum=batch_norm_alpha,
                                                            affine=True,eps=1e-5) 
        layers['nonlin_2'] = later_nonlin()

        
        # Deconv Pool Block1
        layers['unpool'] = first_pool_class(kernel_size=(pool_time_length,1),stride=(pool_stride,1)) 
        layers['deconv_nonlin'] = first_nonlin ()
        if batch_norm:
            layers['bnorm'] = nn.BatchNorm2d(n_filters_conv,momentum=batch_norm_alpha,affine=True,eps=1e-5)
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

################################################# L4 ##################################################
# Deep4AutoEncoder
# Encoder
class Deep4EncoderL4(nn.Module) :
    '''
    Deep4Encoder
    ・Deep4Netデフォルトと流れは同じ
    ・分類器削除
    ・デフォルトでない分岐を削除、使わない引数削除
    when using encoder only, call single() or switch 'is_single' True
    '''
    def __init__(self, 
                in_chans=14, # 電極数
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
                size_check=False,
                is_single=False
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
        layers['dimshuffle'] = Expression(transpose_time_to_spat)
        layers['conv_time'] = nn.Conv2d(in_channels=1, out_channels=n_filters_time, 
                                        kernel_size=(filter_time_length, 1), stride=1)
        layers['conv_spat'] = nn.Conv2d(n_filters_time, n_filters_spat, (1,in_chans), 
                                        stride=(conv_stride,1),bias= not batch_norm) 
        n_filters_conv = n_filters_spat
        if batch_norm:
            layers['bnorm'] = nn.BatchNorm2d(n_filters_conv,momentum=batch_norm_alpha,affine=True,eps=1e-5) 
        layers['conv_nonlin'] = first_nonlin() #  nn.ELU
        layers['pool'] = first_pool_class(kernel_size=(pool_time_length,1),stride=(pool_stride,1), return_indices=True) # nn.MaxPool2d

        # Conv Pool Block2,3,4
        layers['drop_2'] = nn.Dropout(p=drop_prob) # 通すと出力のどれかが0に
        layers['conv_2'] = nn.Conv2d(n_filters_conv, n_filters_2, (filter_length_2, 1),
                                                    stride=(conv_stride, 1), bias=not batch_norm) 
        if batch_norm:
            layers['bnorm_2'] = nn.BatchNorm2d(n_filters_2,momentum=batch_norm_alpha,
                                                                affine=True,eps=1e-5) 
        layers['nonlin_2'] = later_nonlin() # nn.ELU
        layers['pool_2'] = later_pool_class(kernel_size=(pool_time_length, 1),stride=(pool_stride, 1), return_indices=True) # nn.MaxPool2d
        
        layers['drop_3'] = nn.Dropout(p=drop_prob) # 通すと出力のどれかが0に
        layers['conv_3'] = nn.Conv2d(n_filters_2, n_filters_3, (filter_length_3, 1),
                                                    stride=(conv_stride, 1), bias=not batch_norm) 
        if batch_norm:
            layers['bnorm_3'] = nn.BatchNorm2d(n_filters_3,momentum=batch_norm_alpha,
                                                                affine=True,eps=1e-5) 
        layers['nonlin_3'] = later_nonlin() # nn.ELU
        layers['pool_3'] = later_pool_class(kernel_size=(pool_time_length, 1),stride=(pool_stride, 1), return_indices=True) # nn.MaxPool2d

        layers['drop_4'] = nn.Dropout(p=drop_prob) # 通すと出力のどれかが0に
        layers['conv_4'] = nn.Conv2d(n_filters_3, n_filters_4, (filter_length_4, 1),
                                                    stride=(conv_stride, 1), bias=not batch_norm) 
        if batch_norm:
            layers['bnorm_4'] = nn.BatchNorm2d(n_filters_4,momentum=batch_norm_alpha,
                                                                affine=True,eps=1e-5) 
        layers['nonlin_4'] = later_nonlin() # nn.ELU
        layers['pool_4'] = later_pool_class(kernel_size=(pool_time_length, 1),stride=(pool_stride, 1), return_indices=True) # nn.MaxPool2d

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

# Decoder
class Deep4DecoderL4(nn.Module) :
    '''
    Deep4Decoder
    ・Deep4Encoderの逆流
    ・ただしdropoutやbnromの場所を調整
    ・引数はEncoderと同じ
    '''
    def __init__(self, 
                in_chans=14, # 電極数
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
        layers['unpool_4'] = later_pool_class(kernel_size=(pool_time_length, 1),stride=(pool_stride, 1))
        layers['drop_4'] = nn.Dropout(p=drop_prob)
        layers['deconv_4'] = nn.ConvTranspose2d(n_filters_4, n_filters_3, (filter_length_4, 1),
                                                                stride=(conv_stride, 1), bias=not batch_norm) 
        if batch_norm:
            layers['bnorm_4'] = nn.BatchNorm2d(n_filters_3, momentum=batch_norm_alpha,
                                                            affine=True,eps=1e-5) 
        layers['nonlin_4'] = later_nonlin()

        layers['unpool_3'] = later_pool_class(kernel_size=(pool_time_length, 1),stride=(pool_stride, 1))
        layers['drop_3'] = nn.Dropout(p=drop_prob)
        layers['deconv_3'] = nn.ConvTranspose2d(n_filters_3, n_filters_2, (filter_length_3, 1),
                                                                stride=(conv_stride, 1), bias=not batch_norm) 
        if batch_norm:
            layers['bnorm_3'] = nn.BatchNorm2d(n_filters_2, momentum=batch_norm_alpha,
                                                            affine=True,eps=1e-5) 
        layers['nonlin_3'] = later_nonlin()

        layers['unpool_2'] = later_pool_class(kernel_size=(pool_time_length, 1),stride=(pool_stride, 1))
        layers['drop_2'] = nn.Dropout(p=drop_prob)
        layers['deconv_2'] = nn.ConvTranspose2d(n_filters_2, n_filters_conv, (filter_length_2, 1),
                                                                stride=(conv_stride, 1), bias=not batch_norm) 
        if batch_norm:
            layers['bnorm_2'] = nn.BatchNorm2d(n_filters_conv,momentum=batch_norm_alpha,
                                                            affine=True,eps=1e-5) 
        layers['nonlin_2'] = later_nonlin()

        # Deconv Pool Block1
        layers['unpool'] = first_pool_class(kernel_size=(pool_time_length,1),stride=(pool_stride,1)) 
        layers['deconv_nonlin'] = first_nonlin ()
        if batch_norm:
            layers['bnorm'] = nn.BatchNorm2d(n_filters_conv,momentum=batch_norm_alpha,affine=True,eps=1e-5)
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

################################################# AE ##################################################
# Autoencoder
class Deep4AutoEncoder(nn.Module) :
    '''
    Deep4AutoEncoder
    ・Deep4Encoderを改変し、AutoEncoder
    ・Deep4Netから分類機能を削除
    ・その他、機能を一部簡略化
    '''
    def __init__(self, d4_layers=4, **kwargs) :
        '''
        in_chans: 電極数(必須)
        その他引数はEncoder、Decoderで同じ、書くのが面倒なので、**kwargs
        '''
        super().__init__()
        if d4_layers==4 :
            self.encoder = Deep4EncoderL4(**kwargs)
            self.decoder = Deep4DecoderL4(**kwargs)    
        elif d4_layers==3 :
            self.encoder = Deep4EncoderL3(**kwargs)
            self.decoder = Deep4DecoderL3(**kwargs)
        elif d4_layers==2 :
            self.encoder = Deep4EncoderL2(**kwargs)
            self.decoder = Deep4DecoderL2(**kwargs)    
        elif d4_layers==1.5 :
            self.encoder = Deep4EncoderL1_5(**kwargs)
            self.decoder = Deep4DecoderL1_5(**kwargs)     
        elif d4_layers==1 :
            self.encoder = Deep4EncoderL1(**kwargs)
            self.decoder = Deep4DecoderL1(**kwargs)    
        else :
            raise ValueError('n_layers = 1, 1.5, 2, 3, 4 ')
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

      

