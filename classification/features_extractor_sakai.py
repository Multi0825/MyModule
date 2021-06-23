# 坂井関数を簡略化、汎用化、高速化
import numpy as np
from scipy import signal, fftpack, stats

# 窓関数(フレーム分割)
# data: n_ch * n_sample
# win_type: 窓形
# win_size: 窓長
# overlap_rate: オーバーラップ率
def window_data(data, win_type='hamming', win_size=32, overlap_rate=0.5):
    win_datas = []
    overlap_size = int(win_size * overlap_rate)
    start_s = 0
    end_s = start_s + win_size
    win = signal.get_window(win_type, win_size)
    while True:
        win_data = data[:,start_s:end_s] * win
        win_datas.append(win_data)
        start_s = start_s + (win_size - overlap_size)
        end_s = start_s + win_size
        if end_s > data.shape[1]:
            break
    return np.array(win_datas)

# ゼロ交差率(zero crossing rate)
# data: n_win * n_ch * frame_size
def zcr(data):
    return float( np.sum((data[:,:,:-1] * data[:,:,1:] < 0), axis=2) / data.shape[2] )

# Power Spectrum Entropy
# http://int-info.com/PyLearn/PyLearnSIG03.html
# data: n_win * n_ch * frame_size
def pse(data):
    # パワースペクトル算出
    L = data.shape[2] # 信号長
    amp = np.abs(fftpack.fft(data, n=data.shape[-1], axis=2)) # 振幅スペクトル
    power = amp **2 # パワースペクトル
    a = np.sum(power[:,:,1:int(L/2)], axis=2)
    E = -(power[:,:,1:int(L/2)]/a) * np.log2(power[:,:,1:int(L/2)]/a)
    return E

# 5種統計量
# data: n_win * n_ch * frame_size
def stats_feat(data):
	# 1) Root-mean-square (二乗平均平方根)
    feature = np.sqrt(np.mean(data**2, axis=2))
	# 2) ZCR (ゼロ交差率)
    feature = np.concatenate([feature, zcr(data)], axis=2)
	# 3) MA (移動平均)
    feature = np.concatenate([feature, np.mean(data, axis=2)], axis=2)
	# 4) Kurtosis (尖度)
    feature = np.concatenate([feature, stats.kurtosis(data, axis=2)], axis=2)
	# 5) PSE (パワースペクトルエントロピー)
    feature = np.concatenate([feature, pse(data)], axis=2)
    return feature # n_win * n_ch * n_feat


# 3パターンFFT特徴量(パワー、相対位相、両方)
# data: n_win * n_ch * frame_size
# feat_type: magnitude: 振幅スペクトル, phase:相対位相, both: 両方
# sfreq: サンプリング周波数
# using_dim: fft使用次元(1~using_dim)、なぜ6かは不明(written as magic number)
def fft_feat(data, feat_type='both', using_dim=6):
    # FFT計算
    f = fftpack.fft(data, n=data.shape[2], axis=2)
    if feat_type == 'magnitude' or feat_type == 'both':
        f_abs = np.abs(f)
    if feat_type == 'phase' or feat_type == 'both':
        phase = np.angle(f)
        # 基準周波数の位相で正規化 相対的に他の位相を求める
        base_phase = phase[:,:,using_dim] # 各win、ch毎の基本周波数
        relative_phases = [] # 相対位相
        # f_phase[:,:,i] - i/using_dim * base_phase[i]
        for i in range(phase.shape[-1]):
            r_p = phase[:,:, i] - i/using_dim * base_phase
            if i==0:
                relative_phases =  np.expand_dims(r_p, 2)
            else :
                relative_phases =  np.concatenate([relative_phases, np.expand_dims(r_p,2)], axis=2)
        # 位相値を対応する座標値に変換
        coses = np.cos(relative_phases)
        sins = np.sin(relative_phases)
    # 使用次元のみ抽出
    if feat_type == 'magnitude':
        feature = f_abs[:,:,1:using_dim]
    elif feat_type == 'phase':
        feature = np.concatenate([coses[:,:,1:using_dim], sins[:,:,1:using_dim]], axis=2)
    elif feat_type == 'both':
        feature = np.concatenate([f_abs[:,:,1:using_dim], coses[:,:,1:using_dim]],axis=2)
        feature = np.concatenate([feature, sins[:,:,1:using_dim]], axis=2)
    else:
        print('{} does not exists'.format(feat_type))
        pass 
    return feature # n_win * n_ch * n_feat


# 特徴量抽出
# data           : 特徴量を抽出するデータ(n_ch * n_sample)
# ・窓関数
# win_type: 窓形
# frame_size: 窓長(サンプル数)
# overlap_rate: オーバーラップ率
# ・FFT
# feat_type: FFT特徴量(magnitude: 振幅スペクトル, phase:相対位相, both: 両方, None: 統計量)
# using_dim: 使用次元
# ・解析
# n_repeat: 想起回数
# stimuli_frame: 音素を聞かせる刺激のタイミング(s)
# start_frame:  最初の想起音声の開始タイミングを何sずらすか
# slid_frame: 解析区間の移動単位(s)
def extract_feature(data, win_type='hamming', win_size=32, overlap_rate=0.5, 
                    feat_type='both', using_dim=6,
                    n_repeat=5, stimuli_frame=0, start_frame=0, frame_range=3, slid_frame=15) :

    win_data = window_data(data, win_type=win_type, win_size=win_size, overlap_rate=overlap_rate) # n_win*n_ch*frame_size
    # 統計量
    if feat_type==None :
        f = stats_feat(win_data)
    # 窓関数＋FFT
    else :
        f = fft_feat(win_data, feat_type=feat_type, using_dim=using_dim)
    # 抽出した数値にさらに統計解析(平均、標準偏差)を適用、特徴量とす
    feature = []
    current_pos = stimuli_frame + start_frame
    for i in range(n_repeat) :
        mean = np.mean(f[current_pos:current_pos+frame_range,:,:], axis=0).ravel() # 1*n_ch*n_feat
        std = np.std(f[current_pos:current_pos+frame_range,:,:], axis=0).ravel()
        f_ms = np.concatenate([mean, std]) # ex. if feat==both, f_ms: 420{2(mean, std) X 14(n_ch) X 15(amp*5+cos*5+sin*5)}
        feature.extend(f_ms)
        current_pos += slid_frame
    return np.array(feature) 
