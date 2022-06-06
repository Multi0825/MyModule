# 特徴量抽出、ラベルとセットで出力
import numpy as np
from scipy import signal, fft, stats
import pywt
        
# 窓関数を適用し、n_win*n_ch*frame_sizeのデータ作成
# data : n_ch * n_data
# win_type : 窓関数種類(ハミング)
# frame_size : 窓長サンプル点(None: All)
# overlap_rate : フレームサイズに対するオーバーラップ率0~1(None: 0)
def window(data, win_type='hamming', frame_size=None, overlap_rate=0) :
    frame_size = data.shape[1] if frame_size is None else frame_size
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

# FFTにより振幅スペクトルを計算
# data: np.array, dim=1~3,(ex. n_win * n_ch * n_value)
# min_dim, max_dim: min_dim~max_dim次元まで使用 
# *メモ
#   データ点の数は2のべき乗であるべき(らしい)
#   横軸はサンプリング周波数の半分が最大値(左右対称)
#   サンプリング周波数sfreqHz, データ長n点では1/(n/sfreq)Hz刻み
#   scipy.fftではindex0は直流成分(使えない)
#   256点(0~255)あった場合,[1]=[255(-1)],[127]=[129], [128]は1つ(基本的には1~n/2) 
def amp_spectrum(data, n=None, min_dim=1, max_dim=None) :
    n = data.shape[data.ndim-1] if n is None else n
    max_dim = n//2 if max_dim is None else max_dim 
    f_fft = fft.fft(data, n=n, axis=data.ndim-1)
    if data.ndim==3 :
        f_fft = f_fft[:,:,min_dim:max_dim+1]
    elif data.ndim==2 :
        f_fft = f_fft[:,min_dim:max_dim+1]
    else :
        f_fft = f_fft[min_dim:max_dim+1]
    return np.abs(f_fft) # ex. n_win * n_ch * (max_dim+1-min_dim)

# 多重解像度解析(＊詳細係数のみ)
# data : ex. n_ch*n_sample
# wavelet : wavelet type
# level : decomposition level
# mode : 両端の拡張の特性('zero'(default):ゼロ埋め, 'constant':両端を使う, 'symmetric':対称 ...)
# cA: Approximation係数を返すか
def dwt(data, wavelet='db4', level=4, mode='zero', cA=False) :
    coefs = pywt.wavedec(data, wavelet=wavelet, level=level, mode=mode, axis=1)
    cA = int(not cA)
    cDs = [np.array(coefs[i]) for i in range(cA,len(coefs))] # 粗⇔細(cA, cD1,...cDn)
    return cDs # n_dec * n_ch * n_feat 0次元はリスト
    
# 論文参考FFT＋統計量
# using_dim: フーリエ係数使用次元(1~using_dim+1次元)
# n_stats_win: 統計量まとめて計算するフレーム数
# n_stats_overlap: 統計量まとめて計算するオーバーラップ率
def fft_stats(data, win_type, fft_frame_size, overlap_rate, using_dim, n_stats_win, n_stats_overlap) :
    # 窓関数(＊ 全区間を対象(full_full_eeglab_pp)として、想起の前後も含めるのも有か)
    win_data = window(data, win_type=win_type, \
                    frame_size=fft_frame_size, overlap_rate=overlap_rate) # n_win * n_ch * frame_size
    n_win = win_data.shape[0]
    n_ch = win_data.shape[1]

    # FFT
    a_s = amp_spectrum(win_data, max_dim=using_dim) # 1~using_dim次元までの振幅スペクトル
    
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
        range_win_f = a_s[start_win:end_win, :, :]
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

