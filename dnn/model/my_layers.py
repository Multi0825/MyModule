# オリジナル(or Mimic)レイヤー
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class CustomLayer(nn.Module) :
    '''
    任意の計算が可能な層
    '''
    def __init__(self, func, *args, **kwargs):
        '''
        func: 関数
        args: 可変長引数
        kwargs: 可変長キーワード引数
        '''
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def forward(self, x) :
        x = self.func(x, *self.args, **self.kwargs)
        return x


class TimeDistributed(nn.Module) :
    '''
    KerasのTimeDistributedの模倣
    tcapelle Thomas Capelle(https://discuss.pytorch.org/t/timedistributed-cnn/51707/11)
    '''
    def __init__(self, module):
        '''
        module: 対象となるレイヤー
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


class Conv2dAddedBias(nn.Conv2d) :
    '''
    forward時、さらにバイアスが加算できるConv2d
    '''
    def __init__(self, *args, **kwargs):
        # conv宣言時、bias=Falseだとself.bias=None, bias=Trueだとself.bias = torch.tensor()
        kwargs['bias'] = True # biasを作成するため
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor, additional_bias: Tensor = None) -> Tensor:
        # memo
        # bias: out_ch : とにかくout_chの数が重要か　
        new_bias =  additional_bias if self.bias is None else self.bias + additional_bias
        return self._conv_forward(input, self.weight, new_bias)


class ConvTranspose2dAddedBias(nn.ConvTranspose2d) :
    '''
    forward時、さらにバイアスが加算できるDeconv2d
    '''
    def __init__(self, *args, **kwargs) :
        kwargs['bias'] = True # biasを作成するため
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor, additional_bias: Tensor = None, output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')
        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, 
                                              self.kernel_size, self.dilation)  # type: ignore[arg-type]
        # ここまでnn.ConvTranspose2d.forward()をコピー
        new_bias = additional_bias if self.bias is None else self.bias + additional_bias
        return F.conv_transpose2d(input, self.weight, new_bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)