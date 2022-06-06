# 信号処理
import numpy as np
from scipy import signal, fft, stats
import pywt

def window(data, frame_size, win_type='hamming', overlap_rate=0) :
    '''
    窓関数
    data: numpy.ndarray(n_sample x n_ch x n_data)
    win_type: 窓関数タイプ
    frame_size: 窓長(データ点) 
    overlap_rate: オーバーラップ率0~1
    '''
    window = signal.get_window(win_type, frame_size) # 窓関数
    slid_size = int(frame_size * (1-overlap_rate)) # スライド長(オーバーラップ長)
    if (overlap_rate >= 1) or (overlap_rate<0) :
        raise ValueError('0 <= overlap_rate < 1')
    # 窓関数適用
    index = 0
    n_data = data.shape[-1]
    while (index+frame_size) <= n_data :
        data_tmp = np.expand_dims(window * data[:,:,index:index+frame_size], axis=1)
        win_data = data_tmp if index==0 else np.concatenate((win_data, data_tmp), axis=1)
        index += slid_size
    return win_data # n_sample * n_win * n_ch * frame_size

def amp_spectrum(data, n=None, min_dim=1, max_dim=None) :
    '''
    FFTにより振幅スペクトル
    data: np.array(n_sample (x n_win) x n_ch x n_data)
    n: データ長(n点では1/(n/sfreq)Hz刻み)
    min_dim,max_dim: min_dim~max_dim次元まで使用
    '''
    n = data.shape[-1] if n is None else n
    max_dim = n//2 if max_dim is None else max_dim 
    f_fft = fft.fft(data, n=n, axis=data.ndim-1)
    if (data.ndim > 4) or (data.ndim < 3) :
        raise ValueError('data.shape must be (n_sample(,n_win),n_ch,n_data)')
    f_fft = f_fft[:,:,:,min_dim:max_dim+1] if data.ndim==4 else f_fft[:,:,min_dim:max_dim+1]        
    return np.abs(f_fft) # n_sample x n_win x n_ch x (max_dim+1-min_dim)
# FFTメモ
#   データ点の数は2のべき乗であるべき(らしい)
#   横軸はサンプリング周波数の半分が最大値(左右対称)
#   サンプリング周波数sfreqHz, データ長n点では1/(n/sfreq)Hz刻み
#   scipy.fftではindex0は直流成分(使えない)
#   256点(0~255)あった場合,[1]=[255(-1)],[127]=[129], [128]は1つ(基本的には1~n/2) 

def dwt(data, wavelet='db4', level=4, mode='zero', cA=False) :
    '''
    多重解像度解析(DWT)
    data: n_ch x n_data(* ウェーブレット係数の形状が分解レベルで異なるため、1サンプルずつ)
    wavelet: ウェーブレットタイプ
    level: 分解レベル
    mode: 両端の拡張の特性('zero'(default):ゼロ埋め, 'constant':両端を使う, 'symmetric':対称 ...)
    cA: Approximation係数を返すか
    '''
    coefs = pywt.wavedec(data, wavelet=wavelet, level=level, mode=mode, axis=1)
    cA = int(not cA)
    cDs = [np.array(coefs[i]) for i in range(cA,len(coefs))] # 粗⇔細(cA, cD1,...cDn)
    return cDs # n_dec * n_ch * n_feat(0次元はリスト) 

def relative_phases(data, n=None, std_dim=5) :
    '''
    相対位相(坂井)
    基準周波数の位相で正規化、他の位相を相対的に求める？
    data: 信号データ(n_sample x n_win x n_ch x n_data)
    n: データ長
    std_dim: 基準次元(default: 5)
    '''
    n = data.shape[-1] if n is None else n # データ長
    ddim = data.ndim
    fft_f = fft.fft(data, n=n, axis=data.ndim-1) # FFT(最終次元にそって)
    if (ddim > 4) or (ddim < 3) :
        raise ValueError('data.shape must be (n_sample(,n_win),n_ch,n_data)')
    fft_f = fft_f[:,:,:,1:n//2+1] if ddim==4 else fft_f[:,:,:,1:n//2+1]# 対称部分を切り捨て(0も直流成分のため切り捨て)
    phases = np.angle(fft_f) # 位相に変換
    standard_phases = phases[:,:,:,std_dim] if ddim==4 else phases[:,:,std_dim]# 各chの基準周波数の位相
    relative_phases = [] # 相対位相
    for dim in range(phases.shape[-1]) :
        phase = phases[:,:,:,dim] if ddim==4 else phases[:,:,dim]
        r_p = phase - (dim/std_dim * standard_phases) 
        relative_phases = np.expand_dims(r_p, ddim-1) if dim==0 \
                          else np.concatenate((relative_phases, np.expand_dims(r_p,ddim-1)), axis=ddim-1)
    # 位相値を対応する座標値に変換
    sins = np.sin(relative_phases)
    coses = np.cos(relative_phases)
    return sins, coses # n_data x n_ch x ori_data_len/2

if __name__=='__main__' :
    data = np.random.rand(100, 14, 256)
    win_data = window(data, frame_size=128, overlap_rate=0.5) # n_batch x 3 x n_ch x frame_size
    new_data = amp_spectrum(win_data, n=128)
    print(new_data.shape)
    sin,cos = relative_phases(win_data,std_dim=5)
    print(sin.shape)
    print(cos.shape)