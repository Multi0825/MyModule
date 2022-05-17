# DEAP読み込み
# 最低限の機能のみ実装
import pickle as pkl
import numpy as np
import pandas as pd
import mne

def read_deap_dat(dat_fn, sfreq=128, n_ch=32, trial_range=(3,63)) :
    '''
    dat_fn: deap datファイル
    sfreq: サンプリング周波数(128Hz)
    n_ch: 電極数(脳波電極以外33~)
    trial_range: 対象区間(準備期間3s)
    '''
    data_labels = pkl.load(open(dat_fn, 'rb'),encoding='latin1')
    data = data_labels['data'] * 1e-6 # n_epoch(40) x n_ch(40) x n_data(8064)
    labels = data_labels['labels']
    data = data[:,0:n_ch,:] # n_ch=32
    data = data[:,:,sfreq*trial_range[0]:sfreq*trial_range[1]]  # n_data=7680
    return data, labels

class DeapEEG() :
    '''
    csv形式のEEGファイルを読み込む
    '''
    def __init__(self, dat_fn, ch_fn, sfreq=128, n_ch=32, trial_range=(3,63)) :
        '''
        dat_fn: DEAPdatファイル(前処理有)
        ch_fn: 電極ファイル
        sfreq: サンプリング周波数(Default 128Hz)
        n_ch: 電極数(Default 32, 脳波電極以外33~40)
        trial_range: 対象区間(Default 3~63, 準備期間0~3s)
        '''
        # data: n_epoch x n_ch x n_data
        # labels: n_epoch x 4
        self.data, self.labels = read_deap_dat(dat_fn, sfreq, n_ch, trial_range)
        self.sfreq = sfreq
        with open(ch_fn,'r') as f :
            self.ch_names = f.readline().replace('\n','').split(',')
        self.n_ch = len(self.ch_names)
        self.label_names = ['Valence','Arousal','Dominance','Liking']
        self.n_epoch = self.data.shape[0]
        

    # def get_data(self, target_epoch=None, target_chs=None, cutoff=(None, None)):
    #     '''
    #     データ取得
    #     target_epoch : 対象エポック(None:全範囲)
    #     target_chs : 対象電極(None:全範囲)
    #     cutoff: 1エポック内での[min(s), max(s)]を揃える(＊時間のばらつきに対応)
    #     '''
    #     target_chs = self.ch_names if target_chs is None else target_chs 
    #     cutoff[0] = 0 if cutoff[0] is None else cutoff[0]
    #     # エポック指定
    #     if target_epoch is None:
    #         start = int(self.sfreq * cutoff[0])
    #         end = cutoff[1] if cutoff[1] is None else int(self.sfreq * cutoff[1])
    #         data, _ = self.raw[target_chs,start:end] # n_ch*n_sample
    #     else : 
    #         start = int(self.epoch_ranges[target_epoch,0]) + int(self.sfreq * cutoff[0])
    #         end = int(self.epoch_ranges[target_epoch,1])+1 if cutoff[1] is None \
    #               else int(self.epoch_ranges[target_epoch,0]) + int(self.sfreq * cutoff[1])
    #         data, _ = self.raw[target_chs,start:end]
    #     return data
    
    
    def get_split_data(self, target_epochs=None, target_chs=None, cutoff=(None, None)) :
        '''
        データ取得
        3次元(target_epochs(labels) x target_chs x n_sample)に加工ver.
        target_epochs: 対象エポック(None: 全エポック)
        target_chs: 対象電極(None:全範囲)
        cutoff: 1エポック内での[min(s), max(s)]を揃える(＊時間のばらつきに対応)
        '''
        target_epochs = [e for e in range(self.n_epoch)] if target_epochs is None else target_epochs
        target_chs = self.ch_names if target_chs is None else target_chs 
        target_chs = [self.ch_names.index(t_c) for t_c in target_chs] 
        return self.data[target_epochs, target_chs, self.sfreq*cutoff[0]:self.sfreq*cutoff[1]]
        return data


    # def set_data(self, data, target_epoch=None, cutoff=(None, None)) :
    #     '''
    #     データ更新
    #     data : 対象データ
    #     target_epoch : 指定エポック(None:全範囲)
    #     cutoff=[None, None] : epochの範囲を指定
    #     '''
    #     cutoff[0] = 0 if cutoff[0] is None else cutoff[0]
    #     if target_epoch is None :
    #         start = int(self.sfreq * cutoff[0])
    #         end = cutoff[1] if cutoff[1] is None else int(self.sfreq * cutoff[1])
    #         self.raw[:,start:end] = data
    #     else :
    #         start = int(self.epoch_ranges[target_epoch,0]) + int(self.sfreq * cutoff[0])
    #         end = int(self.epoch_ranges[target_epoch,1])+1 if cutoff[1] is None \
    #               else int(self.epoch_ranges[target_epoch,0]) + int(self.sfreq * cutoff[1])
    #         original_data, _= self.raw[:,start:end]
    #         if data.shape == original_data.shape :
    #             self.raw[:,start:end] = data # スライスでは末尾+1


    # def save_data(self, csv_fn) : 
    #     '''
    #     データ保存
    #     csv_fn : データCSVファイル名
    #     '''
    #     # Time
    #     df = pd.DataFrame()
    #     time = self.raw.times
    #     df['Time:{}Hz'.format(int(self.sfreq))] = time
    #     # Epoch
    #     epoch = np.zeros((len(time)))
    #     for e in range(self.n_epoch):
    #         epoch[int(self.epoch_ranges[e,0]):int(self.epoch_ranges[e,1])+1] = e
    #     df['Epoch'] = epoch
    #     # Ch(Data)
    #     ch_datas = self.get_data().T * 1e+6 
    #     df[self.ch_names] = ch_datas
    #     # Label
    #     label = []
    #     for e in range(self.n_epoch) :
    #         label_range = int(self.epoch_ranges[e,1] - self.epoch_ranges[e,0]) + 1
    #         label.extend([self.epoch_labels[e] for i in range(label_range)])
    #     df['Label'] = label
    #     # Stage
    #     stage = []
    #     if len(self.stages)>1 :
    #         for e in range(self.n_epoch) :
    #             for n_stg in range(len(self.stages)) :
    #                 next_stg = self.stages[n_stg+1] if n_stg<len(self.stages)-1 else self.stages[0]
    #                 start = self.stage_starts[self.stages[n_stg]][e]
    #                 end = self.stage_starts[next_stg][e]-1
    #                 stage.extend([self.stages[n_stg] for i in range(end-start+1)])
    #     else :
    #         stage = [self.stages[0] for t in range(len(time))]
    #     df['Stage'] = stage
    #     # CSV出力
    #     df.to_csv(csv_fn, index=False)
    #     print(csv_fn+' has been created')