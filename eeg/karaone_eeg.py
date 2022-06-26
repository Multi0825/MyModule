# KARAONEのデータをEpoc出力CSV形式に合わせたものを読み込む
# EpocEEG変更点
# ラベルファイル読込は無し(CSVは自作のため、必ずラベルを含む)
 
import os
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
import mne

ch_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 
            'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 
            'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 
            'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
            'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 
            'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 
            'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 
            'PO8', 'CB1', 'OZ', 'O2', 'CB2', 'O1'] # 62

class KaraoneEEG():
    '''
    KARAONEデータ(CSVに変換済)を読み込み用クラス
    '''
    def __init__(self, csv_fn, target_stage=None) :
        '''
        csv_fn : KARAONEデータCSV(列: Time:[sfreq]Hz, Epoch, CH1,...CH62, Label, Stage)
        target_stage: None=full, resting, stimuli,thinking, speaking 
        '''
        df = pd.read_csv(csv_fn)
        if target_stage is not None :
            df = df[df['Stage']==target_stage]
            df = df.reset_index(drop=True)
        cols = df.columns
        # mne raw構造体を作成
        # KARAONE有効チャンネル数
        n_ch = 62
        self.n_ch = n_ch
        self.ch_names = ch_names
        data = df.loc[:, self.ch_names].values.T * 1e-6 # mne : V, epoc : μV
        self.sfreq = int(cols[0].split(':')[1].replace('Hz', '')) 
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types='eeg') # ch
        self.raw = mne.io.RawArray(data, info) # mne Raw構造体
        # self.raw.info['sfreq']
        # 
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
        self.stage = [] # ステージ一覧
        self.stage_starts = dict() # ステージの開始点(全てのステージは連続を前提、次のステージの開始点-1が終了点)
                                   # n_stage x epoch
        if 'Stage' in cols :
            _, i_stgs = np.unique(df['Stage'].values, return_index=True)
            self.stages = [df['Stage'][i] for i in sorted(i_stgs)] 
            for stg in self.stages :
                self.stage_starts[stg] = []
                for e in range(self.n_epoch) :
                    self.stage_starts[stg].append(df[(df['Stage']==stg) & (df['Epoch']==e)].index[0])
                self.stage_starts[stg] = np.array(self.stage_starts[stg])
        else :
            self.stage = ['null' for i in range(data.shape[1])]
            self.stage_starts[stg] = [0]
    
    
    def get_data(self, target_epoch=None, target_chs=None, cutoff=[None, None]):
        '''
        データ取得
        target_epoch : 対象エポック(None:全範囲)
        target_chs : 対象電極(None:全範囲)
        cutoff: 1エポック内での[min(s), max(s)]を揃える(＊Karaoneの時間のばらつきに対応)
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
    
    
    def get_split_data(self, target_epochs=None, target_labels=None, target_chs=None, cutoff=[None, None]) :
        '''
        データ取得
        3次元(target_epochs(labels) x target_chs x n_sample)に加工ver.
        target_epochs: 対象エポック(None: 全エポック)
        target_labels: 対象ラベル(target_epochs=Noneのとき)
        target_chs: 対象電極(None:全範囲)
        cutoff: 1エポック内での[min(s), max(s)]を揃える(＊Karaoneの時間のばらつきに対応)
        '''
        if target_epochs is None :
            if target_labels is None :
                target_epochs = [e for e in range(self.n_epoch)]
            else :
                target_epochs = [e for e in range(self.n_epoch) if self.epoch_labels[e] in target_labels]
        data = np.array([self.get_data(target_epoch=e, target_chs=target_chs, cutoff=cutoff) for e in target_epochs])
        return data


    def set_data(self, data, target_epoch=None, cutoff=[None, None]) :
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


    def plot_data(self, target_chs=None, target_epoch=None, tmin=None, tmax=None, scalings=None, block=True, show=True, title=None, out_fn=None):
        '''
        プロット
        target_chs : 対象チャンネル(複数可、None:全範囲)
        target_epoch : 対象エポック(None:全範囲)
        tmin, tmax : ・target_epoch=Noneの時
                     0を基準に時間指定(s),
                   ・target_epoch!=Noneの時
                     対象エポックの開始時間を基準に時間範囲指定(s)、tmin,tmax < 0可
                    ex. tmin=-1, tmax=5 : エポック開始時間の(-1s~+5s)の範囲(tmin < tmax, tmin:None=0, tmax:None=epoch duration)
        scalings : グラフのy軸の大きさ
        block : 画像を消すまで進まない
        show : 画像を表示しない
        out_fn : 指定した場合、画像ファイル出力()
        '''
        title='' if title is None else title
        # mneのフォーマット
        scalings={'eeg':'auto'} if scalings is None else {'eeg':scalings}
        # エポック指定有
        if target_epoch is not None:
            epoch_start = self.raw.times[int(self.epoch_ranges[target_epoch,0])] # エポック開始時間
            epoch_end = self.raw.times[int(self.epoch_ranges[target_epoch,1])] # エポック終了時間
            if tmin is None :
                tmin = 0
            if tmax is None :
                tmax = epoch_end - epoch_start
            # tminはtmax以下でなければならない
            if tmin >= tmax : 
                raise ValueError('time_range[1] must be larger than time_range[0]')
            # 始点、終点調整
            start = epoch_start + tmin # 始点
            if start < 0:
                start = 0
            elif start > self.raw.times[-1] :
                start = self.raw.times[-1]
            end = epoch_start + tmax # 終点
            if end < 0 :
                end = 0
            elif end > self.raw.times[-1] :
                end = self.raw.times[-1]
            duration = end - start # 範囲
            if target_chs != None:
                fig = self.raw.plot(start=start, duration=duration, \
                                    n_channels=target_chs, scalings=scalings, title=title, block=block, show=show)
            else :
                fig = self.raw.plot(start=start, duration=duration, \
                                    scalings=scalings, title=title, block=block, show=show)
            # 画像保存
            if out_fn is not None :
                fig.savefig(out_fn)
                print(out_fn+' has been created')
                #if show :
                fig.clf()
                #fig.close()
        else :
            # エポック指定無
            if (tmin is None) or (tmin<0):
                tmin = 0
            if (tmax is None) or (self.raw.times[0]+tmax):
                tmax = self.raw.times[-1] - self.raw.times[0]
            # tminはtmax以下でなければならない
            if tmin >= tmax :
                raise ValueError('time_range[1] must be larger than time_range[0]') 
            start = self.raw.times[0] + tmin            
            end = self.raw.times[0] + tmax
            duration = end - start
            if target_chs is not None:
                fig = self.raw.plot(start=start, duration=duration, \
                                    n_channels=target_chs, scalings=scalings, title=title, block=block, show=show)
            else :
                fig = self.raw.plot(start=start, duration=duration, \
                                    scalings=scalings, title=title, block=block, show=show)
            # 画像保存
            if out_fn is not None :
                fig.savefig(out_fn)
                print(out_fn+' has been created')
                #if show :
                fig.clf()
                #fig.close()


    def plot_erp(self, target_labels, tmin=None, tmax=None, target_chs=None, labels_dic=None, title=None, show=True, out_fn=None):
        '''
        ERP(事象関連電位)を表示
        target_labels : 事象関連電位対象ラベル
        target_chs : 対象電極
        tmin, tmax : 対象ラベルのエポック開始時間を基準に時間範囲指定(s)、tmin,tmax < 0可
                      ex. tmin=-1, tmax=5 : エポック開始時間の(-1s~+5s)の範囲(tmin < tmax, tmin:None=0,tmax:None=epoch0 duration)
        labels_dic : ラベルとイベント番号の対応辞書(Noneで自動) 
        '''
        # ラベルと番号を対応付け
        if labels_dic is None :
            labels_dic = dict()
            for i in range(len(self.labels)) :
                labels_dic[self.labels[i]] = i 
        else :
            # labels_dicは辞書型でなければならない
            if not(type(labels_dic) is dict):
                raise ValueError('labels_dic must be dict')
        # エポック情報(開始点、ラベル)持つ電極を作成、エポックオブジェクト作成
        events = np.zeros((self.n_epoch, 3)) # n_epoch * [start, 0, label]
        for e in range(self.n_epoch) :
            events[e, 0] = self.epoch_ranges[e, 0]
            events[e, 2] = labels_dic[self.epoch_labels[e]]
        events = events.astype(np.int32) # 整数化
        info = mne.create_info(ch_names=['STI'], sfreq=self.sfreq, ch_types=['stim'])
        stim_data = np.zeros((1, len(self.raw.times)))
        stim_raw = mne.io.RawArray(stim_data, info)
        self.raw.add_channels([stim_raw], force_update_info=True)
        self.raw.add_events(events, stim_channel='STI')
        event_id = [labels_dic[id] for id in target_labels]
        tmin = 0 if tmin is None else tmin
        tmax = self.raw.times[int(self.epoch_ranges[0,1])] if tmax is None else tmax
        # tminはtmax以下でなければならない
        if tmin >= tmax : 
            raise ValueError('tmax must be larger than tmin')
        envoked_no_ref = mne.Epochs(self.raw, events=events.astype(np.int32), event_id=event_id,\
                                    tmin=tmin, tmax=tmax, baseline=None, picks=target_chs).average()
        titles = {'eeg':title} if title is not None else None
        fig = envoked_no_ref.plot(titles=titles, show=show)
        # 画像保存
        if out_fn != None :
            fig.savefig(out_fn)
            print(out_fn+' has been created')
            #fig.clf()
            #fig.close()
        # 電極除外
        self.raw.drop_channels(ch_names=['STI'])


    def resampling(self, new_sfreq,*args,**kwargs) :
        '''
        リサンプリング(epochの範囲等も調整)
        new_sfreq: 変更後サンプリング周波数
        その他: mne.filter.resampleを参照
        '''
        new_data = []
        new_epoch_ranges = np.zeros((self.n_epoch, 2))
        new_stage_starts = {stg:[0 for i in range(self.n_epoch)] for stg in self.stages}
        rate = self.sfreq / new_sfreq
        # エポック毎にresampling
        for e in range(self.n_epoch) :
            new_epoch_ranges[e,0] = new_data.shape[1] if e!=0 else 0
            epoch_data = self.get_data(target_epoch=e)
            if rate >= 1.0 : # Down
                epoch_data = mne.filter.resample(epoch_data, down=rate,*args, **kwargs)
            else : # Up
                epoch_data = mne.filter.resample(epoch_data, up=rate,*args, **kwargs)
            new_data = np.concatenate([new_data, epoch_data], axis=1) if e!=0 else epoch_data
            new_epoch_ranges[e,1] = new_data.shape[1]-1
            for stg in self.stages :
                start = int((self.stage_starts[stg][e]-self.epoch_ranges[e,0]) / rate)
                new_stage_starts[stg][e] = start+new_epoch_ranges[e,0] if self.stages.index(stg)!=0 else new_epoch_ranges[e,0]
        self.sfreq = new_sfreq
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types='eeg') # ch
        self.raw = mne.io.RawArray(new_data, info) # mne Raw構造体
        self.epoch_ranges = new_epoch_ranges
        self.stage_starts = new_stage_starts
        
    
    # 基準値減算したデータ取得(各エポック毎、各電極の平均値をオリジナルから引く)
    def remove_baseline(self) :
        new_data = np.empty((self.n_ch,0))
        for e in range(self.n_epoch) :
            epoch_data = self.get_data(target_epoch=e)
            epoch_means = np.mean(epoch_data, axis=1) # エポックの各チャンネル毎の平均値
            for ch in range(self.n_ch) :
                epoch_data[ch] = epoch_data[ch] - epoch_means[ch]    
            new_data = np.concatenate([new_data, epoch_data], axis=1)
        return new_data
    

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
                for n_stg,stg in enumerate(self.stages) :
                    # resting,0 -> ... -> speaking,0 -> resting, 1 -> ...
                    next_stg = self.stages[(n_stg+1)%len(self.stages)] 
                    start = self.stage_starts[stg][e]
                    end = self.stage_starts[next_stg][e]-1 if n_stg!=(len(self.stages)-1) else self.epoch_ranges[e,1]
                    stage.extend([stg for i in range(int(end-start+1))])
        else :
            stage = [self.stages[0] for t in range(len(time))]
        df['Stage'] = stage
        # CSV出力
        df.to_csv(csv_fn, index=False)
        print(csv_fn+' has been created')

        

        


