# 特徴量抽出、ラベルとセットで出力
import numpy as np
from scipy import signal, fftpack, stats
import pywt
        
# 窓関数を適用し、n_window*n_ch*frame_sizeのデータ作成
# data : n_ch * n_sample
# win_type : 窓関数種類(ハミング)
# frame_size : 窓長(None: All)
# overlap_rate : フレームサイズに対するオーバーラップ率0~1(None: 0)
def window(data, win_type='hamming', frame_size=None, overlap_rate=0) :
    if frame_size==None :
        frame_size = data.shape[1] # 全点
    window = signal.get_window(win_type, frame_size) # 窓関数
    slid_size = int(frame_size * (1-overlap_rate)) # スライド長(オーバーラップ長)
    if slid_size <= 0 :
        slid_size = 1
    elif slid_size > frame_size:
        slid_size = frame_size
    # 窓関数適用
    win_data = [] # n_win* n_ch * frame_size
    index = 0
    n_sample = len(data[0])
    while (index+frame_size) <= n_sample :
        win_data.append(window * data[:,index:index+frame_size])
        index += slid_size
    win_data = np.array(win_data)
    return win_data


# データの中心差分(両端は前方差分)を取得
# data : ndarray (n_win * n_ch * n_f)
def delta(data) :
    delta = []
    n_win = data.shape[0] 
    delta.append(data[1,:,:] - data[0,:,:])
    for w in range(1, n_win-1) :
        delta.append(data[w+1,:,:] - data[w-1,:,:])
    delta.append(data[-1,:,:] - data[-2,:,:])
    return np.array(delta)


# 統計量(最大、最小、(最大±最小)、平均、分散、標準偏差、尖度、歪度)取得
stats_func = {'max':np.max, 'min':np.min, 'max+min':lambda x,axis:np.max(x, axis=axis)+np.min(x,axis=axis), \
              'max-min':lambda x,axis:np.max(x, axis=axis)-np.min(x,axis=axis), 'mean':np.mean, 'var':np.var, 'std':np.std,\
              'kurt':stats.kurtosis, 'skew':stats.skew}
use_stats = {'max':True, 'min':True, 'max+min':True, \
             'max-min':True, 'mean':True, 'var':True, 'std':True,\
             'kurt':True, 'skew':True} # 統計量を使用するか
# data : ndarray(n_win(n_dec) * n_ch * n_sample)
def stats_features(data, axis=2) :
    n_win = len(data)
    n_ch = data[0].shape[0]
    f = []
    for w in range(0,n_win) :
        i = 0
        w_f = []
        for sf_name in stats_func.keys() :
            if i == 0 :
                if use_stats[sf_name] :
                    w_f = stats_func[sf_name](data[w], axis=1).reshape(n_ch, 1)
                    i += 1
            else :
                if use_stats[sf_name] :
                    w_f = np.concatenate([w_f, stats_func[sf_name](data[w], axis=1).reshape(n_ch,1)], axis=1) 
        f.append(w_f)
    return np.array(f)

# 使用非推奨
def stats_features2(data, axis=2) :
    dim_0 = len(data)
    dim_1 = data[0].shape[0]
    dim_2 = data[0].shape[1]
    f = []
    i = 0
    for sf_name in stats_func.keys() :
        if i == 0 :
            if use_stats[sf_name] :
                f = stats_func[sf_name](data, axis=axis)
                if axis==2 :
                    f = f.reshape(dim_0, dim_1, 1)
                elif axis == 0 :
                    f = f.reshape(1, dim_1, dim_2)
                else : 
                    raise ValueError('Sorry axis=1 is not implemented') 
                i += 1
        else :
            if use_stats[sf_name] :
                stats_f = stats_func[sf_name](data, axis=axis)
                if axis==2 :
                    stats_f = stats_f.reshape(dim_0, dim_1, 1)
                elif axis == 0 :
                    stats_f = stats_f.reshape(1, dim_1, dim_2)
                else : 
                    raise ValueError('Sorry Only Implemented axis=2 or axis=0') 
                f = np.concatenate([f, stats_f], axis=axis)
    return np.array(f)

# FFT(複素数のためr**2+i**2)
# data : ndarray(n_win * n_ch * n_sample)
def fft(data) :
    f_fft = fftpack.fft(data, n=data.shape[2], axis=2)
    return np.abs(f_fft)

# 多重解像度解析(＊詳細係数のみ、次元0がリスト)
# data : ndarrry(n_ch*n_sample)
# wavelet : wavlet
# level : decomposition level
# mode : 両端の拡張の特性('zero'(default):ゼロ埋め, 'constant':両端を使う, 'symmetric':対称 ...)
def dwt(data, wavelet='db4', level=4, mode='zero') :
    coefs = pywt.wavedec(data, wavelet=wavelet, level=level, mode=mode, axis=1)
    cDs = [np.array(coefs[i]) for i in range(1,len(coefs))] # 詳細係数のみ(粗⇔細)
    return cDs # n_dec * n_ch * n_feature
    
# 論文参考FFT＋統計量(不完全)
# using_dim: フーリエ係数使用次元(1~using_dim+1次元)
# n_stats_win: 統計量まとめて計算するフレーム数
# n_stats_overlap: 統計量まとめて計算するオーバーラップ率
def fft2(data, win_type, fft_frame_size, overlap_rate, using_dim, n_stats_win, n_stats_overlap) :
    # 窓関数(＊ 全区間を対象(full_full_eeglab_pp)として、想起の前後も含めるのも有か)
    win_data = window(data, win_type=win_type, \
                    frame_size=fft_frame_size, overlap_rate=overlap_rate) # n_win * n_ch * frame_size
    n_win = win_data.shape[0]
    n_ch = win_data.shape[1]

    # FFT
    f_fft = fft(win_data) # 7,11,15,31,47,63,79, * 3 * 32
    low_dim_f = f_fft[:,:,1:using_dim+1] # 下からusing_dim次元まで使う 
    
    # n_win * n_ch * using_dimのフーリエ係数に対して、n_stats_win毎、各電極毎に統計量をとる
    start_win = 0 # 
    end_win = 0
    if n_stats_win > n_stats_overlap :
        slid_win = n_stats_win - n_stats_overlap
    else : 
        raise ValueError('n_stats_win must be larger than n_stats_overlap')
    stats_func = {'max':np.max, 'min':np.min, 'max+min':lambda x:np.max(x)+np.min(x), \
            'max-min':lambda x:np.max(x)-np.min(x), 'mean':np.mean, 'var':np.var, 'std':np.std,\
                'kurt':lambda x:stats.kurtosis(x, axis=None), 'skew': lambda x:stats.skew(x,axis=None)}
    win_stats_f = []
    while end_win < n_win+1 :
        end_win = start_win + n_stats_win
        range_win_f = low_dim_f[start_win:end_win, :, :]
        ch_stats_f = []
        for ch in range(0, n_ch) :
            ch_win_f = range_win_f[:,ch,:]
            stats_f = [sf(ch_win_f) for sf in stats_func.values()]
            stats_f = np.array(stats_f).reshape(1, -1)
            if ch == 0:
                ch_stats_f = stats_f
            else :
                ch_stats_f = np.concatenate([ch_stats_f, stats_f],axis=0)
        ch_stats_f = ch_stats_f.reshape(1, n_ch, -1)
        if start_win == 0 :
            win_stats_f = ch_stats_f
        else :
            win_stats_f = np.concatenate([win_stats_f, ch_stats_f])
        start_win += slid_win
    # 統計値の差分は
    if win_stats_f.shape[0] > 1 :
        d_f = delta(win_stats_f)
        dd_f = delta(d_f)
        features = np.concatenate([win_stats_f, d_f, dd_f], axis=2) 
    else :
        features = win_stats_f
    return features