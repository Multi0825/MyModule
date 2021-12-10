# Yue Yao, Deep Feature Learning for EEG Recording Using Autoencoders
# image-wise Autoencoder
# 参考 https://qiita.com/shushin/items/3f35a9a4200d7be74dc9
# 中間の形が合わん
# * 重みをEとDで共有するパターンもあった
import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder
class ImageWiseEncoder(nn.Module) :
    def __init__(self, 
                 conv1_param, pool1_param, edrop1_param, 
                 conv2_param, pool2_param, edrop2_param, 
                 conv3_param):
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

    def forward(self, x, size_check=False) :
        pool_indices = [] # decoderのunpoolで使用する
        for name,layer in self.layers.items() :
            if 'pool' in name :
                x, indices = layer(x)
                pool_indices.append(indices)
            else :
                x = layer(x)
            if size_check :
                print('{}:{}'.format(name, x.size()))

        return x, pool_indices
    
    # クラス分類に転用する場合等のための層追加
    def add_layer(self, key, layer) :
        self.layers[key] = layer
        
# Decoder
# 動作 参考 https://qiita.com/elm200/items/621b410e69719df0e6f4
class ImageWiseDecoder(nn.Module) :
    def __init__(self, 
                 deconv1_param, unpool1_param, ddrop1_param,
                 deconv2_param, unpool2_param, ddrop2_param, 
                 deconv3_param):
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

    def forward(self, x, pool_indices, size_check=False) :
        p_i = 1 # poolのカウント(逆から使用)
        for name,layer in self.layers.items() :
            if 'unpool' in name :
                x = layer(x, pool_indices[-p_i])
                p_i += 1
            else :
                x = layer(x)
            if size_check :
                print('{}:{}'.format(name, x.size()))
        return x


# AutoEncoder
class ImageWiseAutoEncoder(nn.Module) :
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
                 deconv3_param=None) :
        super().__init__()
        # EncoderとDecoderを合わせる
        if deconv1_param is None :
            deconv1_param = conv3_param.copy() # 辞書型はイミュータブルなのでcopy必須
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
                                        conv3_param=conv3_param)
        self.decoder = ImageWiseDecoder(deconv1_param=deconv1_param, unpool1_param=unpool1_param, ddrop1_param=ddrop1_param,
                                        deconv2_param=deconv2_param, unpool2_param=unpool2_param, ddrop2_param=ddrop2_param,
                                        deconv3_param=deconv3_param)

    def forward(self, x, size_check=False) :
        if size_check :
            print('Encoder')
        x, pool_indices = self.encoder(x, size_check)
        if size_check :
            print('Decoder')
        x = self.decoder(x, pool_indices, size_check)
        return x


if __name__=='__main__' :
    # test
    input = torch.rand((100, 3, 32, 32)) # n_sample x n_color x H x W
    model = ImageWiseAutoEncoder()
    output = model(input, True)


