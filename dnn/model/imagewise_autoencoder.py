# Yue Yao, Deep Feature Learning for EEG Recording Using Autoencoders
# image-wise Autoencoder
# 参考 https://qiita.com/shushin/items/3f35a9a4200d7be74dc9

import torch
import torch.nn as nn

# Encoder
class ImageWiseEncoder(nn.Module) :
    '''
    Image-wise Encoder
    when using encoder only, call single() or switch 'is_single' True
    '''
    def __init__(self, 
                 conv1_param, pool1_param, edrop1_param, 
                 conv2_param, pool2_param, edrop2_param, 
                 conv3_param, size_check=False, is_single=False):
        super().__init__()

        layers = {}
        layers['conv1'] = nn.Conv2d(**conv1_param) # h_out = (h_in + 2pad - kernel)/stride + 1
        layers['pool1'] = nn.MaxPool2d(**pool1_param, return_indices=True) # indicesはunpoolで使用
        layers['relu1'] = nn.ReLU()
        layers['drop1'] = nn.Dropout(**edrop1_param)
        
        layers['conv2'] = nn.Conv2d(**conv2_param)
        layers['pool2'] = nn.MaxPool2d(**pool2_param, return_indices=True)
        layers['relu2'] = nn.ReLU()
        layers['drop2'] = nn.Dropout(**edrop2_param)

        layers['conv3'] = nn.Conv2d(**conv3_param)
        layers['relu3'] = nn.ReLU()
        self.layers =  nn.ModuleDict(layers)
        self.size_check = size_check
        self.is_single = is_single # 単体で使う

    def forward(self, x) :
        # Encoder単体
        if self.is_single:
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
            pool_indices = [] # decoderのunpoolで使用
            pool_size = [] # decoderのunpoolのサイズ合わせで使用
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
# 動作 参考 https://qiita.com/elm200/items/621b410e69719df0e6f4
class ImageWiseDecoder(nn.Module) :
    '''
    Image-wise Decoder
    '''
    def __init__(self, 
                 deconv1_param, unpool1_param, ddrop1_param,
                 deconv2_param, unpool2_param, ddrop2_param, 
                 deconv3_param, size_check=False):
        super().__init__()
        # 下記の構造は論文を参考に
        layers = {}
        layers['deconv1'] = nn.ConvTranspose2d(**deconv1_param) # h_out = (h_in - 1)xstride-2pad+kernel+outpad
        layers['unpool1'] = nn.MaxUnpool2d(**unpool1_param)
        layers['relu1'] = nn.ReLU()
        layers['drop1'] = nn.Dropout(**ddrop1_param)
        
        layers['deconv2'] = nn.ConvTranspose2d(**deconv2_param)
        layers['unpool2'] = nn.MaxUnpool2d(**unpool2_param)
        layers['relu2'] = nn.ReLU()
        layers['drop2'] = nn.Dropout(**ddrop2_param)

        layers['deconv3'] = nn.ConvTranspose2d(**deconv3_param)
        self.layers =  nn.ModuleDict(layers)
        self.size_check = size_check

    def forward(self, x, pool_indices, pool_size) :
        p_i = -1 # poolのカウント(逆から使用)
        for name,layer in self.layers.items() :
            if 'unpool' in name :
                x = layer(x, indices=pool_indices[p_i], output_size=pool_size[p_i])
                p_i -= 1
            else :
                x = layer(x)
            if self.size_check :
                print('{}:{}'.format(name, x.size()))
        return x


# AutoEncoder
class ImageWiseAutoEncoder(nn.Module) :
    '''
    Image-wise AutoEncoder
    '''
    def __init__(self, 
                 # E
                 conv1_param={'in_channels':3, 'out_channels':16, 'kernel_size':(3,3), 'stride':(1,1)}, 
                 pool1_param={'kernel_size':(2,2) , 'stride':(1,1)},
                 edrop1_param={'p':0.25}, 
                 conv2_param={'in_channels':16, 'out_channels':16, 'kernel_size':(3,3), 'stride':(1,1)}, 
                 pool2_param={'kernel_size':(2,2) , 'stride':(1,1)}, 
                 edrop2_param={'p':0.25}, 
                 conv3_param={'in_channels':16, 'out_channels':16, 'kernel_size':(3,3), 'stride':(1,1)},
                 # D
                 deconv1_param=None, unpool1_param=None, ddrop1_param=None,
                 deconv2_param=None, unpool2_param=None, ddrop2_param=None, 
                 deconv3_param=None, 
                 size_check=False) :
        super().__init__()
        # EncoderとDecoderを合わせる
        if deconv1_param is None :
            deconv1_param = conv3_param.copy() # 辞書型はミュータブルなのでcopy必須
            deconv1_param['in_channels'] = conv3_param['out_channels']
            deconv1_param['out_channels'] = conv3_param['in_channels']
        unpool1_param = pool2_param.copy() if unpool1_param is None else unpool1_param
        ddrop1_param = edrop2_param.copy() if ddrop1_param is None else ddrop1_param
        if deconv2_param is None :
            deconv2_param = conv2_param.copy()
            deconv2_param['in_channels'] = conv2_param['out_channels']
            deconv2_param['out_channels'] = conv2_param['in_channels']
        unpool2_param = pool1_param.copy() if unpool2_param is None else unpool2_param
        ddrop2_param = edrop1_param.copy() if ddrop2_param is None else ddrop2_param
        if deconv3_param is None :
            deconv3_param = conv1_param.copy()
            deconv3_param['in_channels'] = conv1_param['out_channels']
            deconv3_param['out_channels'] = conv1_param['in_channels']

        self.encoder = ImageWiseEncoder(conv1_param=conv1_param, pool1_param=pool1_param, edrop1_param=edrop1_param, 
                                        conv2_param=conv2_param, pool2_param=pool2_param, edrop2_param=edrop2_param,
                                        conv3_param=conv3_param, size_check=size_check)
        self.decoder = ImageWiseDecoder(deconv1_param=deconv1_param, unpool1_param=unpool1_param, ddrop1_param=ddrop1_param,
                                        deconv2_param=deconv2_param, unpool2_param=unpool2_param, ddrop2_param=ddrop2_param,
                                        deconv3_param=deconv3_param, size_check=size_check)
        self.size_check = size_check

    def forward(self, x) :
        '''
        x: n_sample x D x H x W
        '''
        if self.size_check :
            print('Encoder')
        x, pool_indices, pool_size = self.encoder(x)
        if self.size_check :
            print('Decoder')
        x = self.decoder(x, pool_indices, pool_size)
        return x

# Test
if __name__=='__main__' :
    
    input = torch.rand((100, 3, 32, 32)) # n_sample x n_color x H x W
    model = ImageWiseAutoEncoder(size_check=True)
    output = model(input)


