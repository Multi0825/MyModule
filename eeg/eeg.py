# EEG統合
# EpocEEG、KaraoneEEGを簡略化、色々なデータに
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
import mne

class EEG() :
    '''
    csv形式のEEGファイルを読み込む
    '''
    def __init__(self, csv_fn, n_ch, target_stage=None) :
        '''
        csv_fn : EEGデータCSV(列: Time:[sfreq]Hz, Epoch, CH1,...CHn_ch, Label, Stage)
        n_ch: 電極数
        target_stage: None=full 
        '''
        df = pd.read_csv(csv_fn)
        if target_stage is not None :
            df = df[df['Stage']==target_stage]
            df = df.reset_index(drop=True)
        cols = df.columns
        # mne raw構造体を作成
        self.n_ch = n_ch
        self.ch_names = cols.to_list()[2:2+self.n_ch] 
        data = df.loc[:, self.ch_names].values.T * 1e-6 # mne : V, epoc : μV
        self.sfreq = int(cols[0].split(':')[1].replace('Hz', '')) 
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types='eeg') # ch
        self.raw = mne.io.RawArray(data, info) # mne Raw構造体
        
        # エポックのラベルを読み込み
        self.n_epoch = int(np.max(df.loc[:,'Epoch'].values)) + 1 # エポック数
        self.epoch_labels = [df[df['Epoch']==e].iloc[-1]['Label'] for e in range(self.n_epoch)]
        self.labels = sorted(set(self.epoch_labels))
        
        # エポックの範囲(サンプル番号)
        self.epoch_ranges = np.zeros((self.n_epoch, 2)) # エポックの範囲(開始点, 終了点)
        for e in range(self.n_epoch) :
            self.epoch_ranges[e,0] = df[df['Epoch']==e].index[0]
            self.epoch_ranges[e,1] = df[df['Epoch']==e].index[-1] 
        
        # ステージ
        self.stages = list(np.unique(df['Stage'].values))
        self.stage_starts = dict() # ステージの開始点 stg x [start_e0, ... start_eN](全てのステージは連続を前提、次のステージの開始点-1が終了点)
        for stg in self.stages :
            self.stage_starts[stg] = []
            for e in range(self.n_epoch) :
                self.stage_starts[stg].append(df[(df['Stage']==stg) & (df['Epoch']==e)].index[0])


    def get_data(self, target_epoch=None, target_chs=None, cutoff=(None, None)):
        '''
        データ取得
        target_epoch : 対象エポック(None:全範囲)
        target_chs : 対象電極(None:全範囲)
        cutoff: 1エポック内での[min(s), max(s)]を揃える(＊時間のばらつきに対応)
        '''
        target_chs = self.ch_names if target_chs is None else target_chs 
        cutoff[0] = 0 if cutoff[0] is None else cutoff[0]
        # エポック指定
        if target_epoch is None:
            start = int(self.sfreq * cutoff[0])
            end = cutoff[1] if cutoff[1] is None else int(self.sfreq * cutoff[1])
            data, _ = self.raw[target_chs,start:end] # n_ch*n_sample
        else : 
            start = int(self.epoch_ranges[target_epoch,0]) + int(self.sfreq * cutoff[0])
            end = int(self.epoch_ranges[target_epoch,1])+1 if cutoff[1] is None \
                  else int(self.epoch_ranges[target_epoch,0]) + int(self.sfreq * cutoff[1])
            data, _ = self.raw[target_chs,start:end]
        return data
    
    
    def get_split_data(self, target_epochs=None, target_labels=None, target_chs=None, cutoff=(None, None)) :
        '''
        データ取得
        3次元(target_epochs(labels) x target_chs x n_sample)に加工ver.
        target_epochs: 対象エポック(None: 全エポック)
        target_labels: 対象ラベル(target_epochs=Noneのとき)
        target_chs: 対象電極(None:全範囲)
        cutoff: 1エポック内での[min(s), max(s)]を揃える(＊時間のばらつきに対応)
        '''
        if target_epochs is None :
            if target_labels is None :
                target_epochs = [e for e in range(self.n_epoch)]
            else :
                target_epochs = [e for e in range(self.n_epoch) if self.epoch_labels[e] in target_labels]
        data = np.array([self.get_data(target_epoch=e, target_chs=target_chs, cutoff=cutoff) for e in target_epochs])
        return data


    def set_data(self, data, target_epoch=None, cutoff=(None, None)) :
        '''
        データ更新
        data : 対象データ
        target_epoch : 指定エポック(None:全範囲)
        cutoff=[None, None] : epochの範囲を指定
        '''
        cutoff[0] = 0 if cutoff[0] is None else cutoff[0]
        if target_epoch is None :
            start = int(self.sfreq * cutoff[0])
            end = cutoff[1] if cutoff[1] is None else int(self.sfreq * cutoff[1])
            self.raw[:,start:end] = data
        else :
            start = int(self.epoch_ranges[target_epoch,0]) + int(self.sfreq * cutoff[0])
            end = int(self.epoch_ranges[target_epoch,1])+1 if cutoff[1] is None \
                  else int(self.epoch_ranges[target_epoch,0]) + int(self.sfreq * cutoff[1])
            original_data, _= self.raw[:,start:end]
            if data.shape == original_data.shape :
                self.raw[:,start:end] = data # スライスでは末尾+1


    def save_data(self, csv_fn) : 
        '''
        データ保存
        csv_fn : データCSVファイル名
        '''
        # Time
        df = pd.DataFrame()
        time = self.raw.times
        df['Time:{}Hz'.format(int(self.sfreq))] = time
        # Epoch
        epoch = np.zeros((len(time)))
        for e in range(self.n_epoch):
            epoch[int(self.epoch_ranges[e,0]):int(self.epoch_ranges[e,1])+1] = e
        df['Epoch'] = epoch
        # Ch(Data)
        ch_datas = self.get_data().T * 1e+6 
        df[self.ch_names] = ch_datas
        # Label
        label = []
        for e in range(self.n_epoch) :
            label_range = int(self.epoch_ranges[e,1] - self.epoch_ranges[e,0]) + 1
            label.extend([self.epoch_labels[e] for i in range(label_range)])
        df['Label'] = label
        # Stage
        stage = []
        if len(self.stages)>1 :
            for e in range(self.n_epoch) :
                for n_stg in range(len(self.stages)) :
                    next_stg = self.stages[n_stg+1] if n_stg<len(self.stages)-1 else self.stages[0]
                    start = self.stage_starts[self.stages[n_stg]][e]
                    end = self.stage_starts[next_stg][e]-1
                    stage.extend([self.stages[n_stg] for i in range(end-start+1)])
        else :
            stage = [self.stages[0] for t in range(len(time))]
        df['Stage'] = stage
        # CSV出力
        df.to_csv(csv_fn, index=False)
        print(csv_fn+' has been created')