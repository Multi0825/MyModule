# 途中
# EEG Base
# EEG CSVファイル(EmotivEpoc産＋オリジナル要素)を読み込む
# Time:xxxHz, 
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
import mne

class _EEG_Base() :
    '''
    csv_fn: CSVファイル名(以下の列名、電極名、電極数は任意, Label,Stageは無くても)
            Time:xxxHz, Epoch, Ch1, ..., ChN, (Label, Stage)
    n_ch: 電極数
    labels_fn: ラベルを外部ファイルとする場合
    '''
    def __init__(self, csv_fn, n_ch, labels_fn=None, target_stage=None) -> None:
        df = pd.read_csv(csv_fn)
        # 特定ステージのみ選択
        if target_stage is not None :
            df = df[df['Stage']==target_stage]
            df = df.reset_index(drop=True)
        cols = df.columns
        # 基本情報
        self.n_ch = n_ch # 電極数
        self.ch_names = cols.to_list()[2:2+self.n_ch] # 電極名
        self.sfreq = int(cols[0].split(':')[1].replace('Hz', '')) # サンプリング周波数
        self.n_epoch = int(np.max(df.loc[:,'Epoch'].values)) + 1 # エポック数
        # mne RawArrayで信号を保持
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types='eeg') 
        data = df.loc[:, self.ch_names].values.T * 1e-6 # 信号(epoch:μV -> mne:V)
        self.raw = mne.io.RawArray(data, info) # mne Raw構造体
        # labels: ラベルの種類, epoch_labels: エポックのラベルを順番に保持
        self.epoch_labels = [df[df['Epoch']==e].iloc[-1]['Label'] for e in range(self.n_epoch)]
        self.labels = sorted(set(self.epoch_labels))
        pass



