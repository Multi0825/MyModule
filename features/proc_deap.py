import numpy as np
import scipy.stats as stats
from ..utils.my_util import no_duplicated_randint

# 電極名:(y,x) 左上が(0,0)
ch_map = {'FP1':(0,3), 'AF3':(1,3), 'F7':(2,0), 'F3':(2,2), 'FC1':(3,3), 'FC5':(3,1), 'T7':(4,0), 'C3':(4,2), 
          'CP1':(5,3), 'CP5':(5,1), 'P7':(6,0), 'P3':(6,2), 'PZ':(6,4), 'PO3':(7,3), 'O1':(8,3), 'OZ':(8,4), 
          'O2':(8,5), 'PO4':(7,5), 'P4':(6,6), 'P8':(6,8),'CP6':(5,7), 'CP2':(5,5), 'C4':(4,6), 'T8':(4,8), 
          'FC6':(3,7), 'FC2':(3,5), 'F4':(2,6), 'F8':(2,8), 'AF4':(1,5), 'FP2':(0,5), 'FZ':(2,4), 'CZ':(4,4)}

def subsampling(data, labels, n_select=5, n_split=6,
                     valence=(None,5), arousal=(None,5), dominance=(None,None), liking=(None,None), 
                     random_seed=None) :
    '''
    DEAP Subsampling
    ラベルに対応するエポックをn_select個選択(未満は無し)、n_split個に分割
    (n_select x n_split) x n_ch x n_data//n_split
    data: n_epoch x n_ch x n_data
    labels: n_epoch x 4(Valence, Arousal, Dominance, Liking)
    n_select: 選択するエポックの数
    n_split: 分割個数
    valence, arousal, dominance, liking: (min, max)該当するもののみ
    random_seed: シード値
    '''
    n_epoch = data.shape[0]
    n_data = data.shape[2]
    subsampled_data = []
    subsampled_labels =[]
    # 条件を満たすエポックを選択
    target_epochs = []
    for e in range(n_epoch) :
        is_target = True
        for i, state in enumerate((valence, arousal, dominance, liking)) :
            min = -np.inf if state[0] is None else state[0]
            max = np.inf if state[1] is None else state[1]
            if not (labels[e,i]>min) & (labels[e,i]<max) :
                is_target = False
                break
        if is_target :
            target_epochs.append(e)
    # n_select個以上が該当する場合、n_select個選択
    if len(target_epochs) >= n_select :
        target_epochs = no_duplicated_randint(0, len(target_epochs), n_select, random_seed=random_seed)# 乱数5つ
        new_n_data = n_data//n_split
        for e in target_epochs : 
            subsampled_data.extend([data[e,:,new_n_data*n_s:new_n_data*(n_s+1)] for n_s in range(n_split)])
            subsampled_labels.extend([labels[e,:] for n_s in range(n_split)])
    return np.array(subsampled_data), np.array(subsampled_labels)

def channels_mapping(data, len_seq=10, ch_names=list(ch_map.keys()), sfreq=128, mesh_size=9) :
        '''
        時間単位でメッシュに32電極をマッピング(n_sample x len_seq x sfreq x mesh_size x mesh_size)
        data: n_sample x n_ch x n_data
        len_seq: 時間長(min=1, max=n_data/sfreq)
        ch_names: 電極名リスト
        sfreq: サンプリング周波数
        mesh_size: メッシュ一辺
        '''
        n_sample = data.shape[0]
        # メッシュ単位の標準化(=ch単位)
        data = stats.zscore(data, axis=1)
        # マッピング
        meshes = np.zeros((n_sample, len_seq, sfreq, mesh_size, mesh_size)) # 論文と次元の順番が違う(Keras<->Torch) 
        for n, ch in enumerate(ch_names) :
            x = ch_map[ch][1]
            y = ch_map[ch][0]
            for l_s in range(len_seq) :
                meshes[:,l_s,:,y,x] = data[:,n,sfreq*l_s : sfreq*(l_s+1)]
        return meshes