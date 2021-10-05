import os
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
import mne

# EmotivEpoc出力のCSVファイル操作関数
# CSVファイルにラベルを追加
def include_labels(csv_fn, labels_fn, out_fn=None) :
    df = pd.read_csv(csv_fn)
    cols = df.columns.to_list()
    with open(labels_fn, mode='r') as f:
        labels = [l.replace('\n','') for l in f]
    print('Reading '+ csv_fn + ' & ' + labels_fn)
    n_epoch = np.max(df.loc[:,'Epoch'].values)+1
    # CSVのエポック数とラベルファイルのラベル数(行数)は同じでなければならない
    if len(labels) != n_epoch :
        raise ValueError('Number of labels must be same as number of epochs')
    # 対応したエポックにラベルを
    epoch_labels = []
    for e in range(n_epoch) :
        label_range = len(df[df['Epoch']==e].index)
        epoch_labels.extend([labels[e] for i in range(label_range)])
    # 挿入
    if 'Label' in cols :
        df = df.drop(columns='Label')
        cols = df.columns.to_list()
    df.insert(len(cols), 'Label', epoch_labels) # 最後に追加
    # CSV出力
    out_fn = csv_fn if out_fn is None else out_fn
    df.to_csv(out_fn, index=False)
    print(out_fn+' has been created')

# 段階で分けられたCSVデータを時系列順に並び替えて結合し、出力
def sort_csv(csv_fns, out_fn=None):
    dfs = [] # CSVs
    for csv_fn in csv_fns :
        print('Reading '+csv_fn)
        dfs.append(pd.read_csv(csv_fn))
    # 1つ目のCSVを基準にエポック数、ヘッダを
    n_epoch = np.max(dfs[0].loc[:,'Epoch'].values)+1
    columns = dfs[0].columns
    full_df = pd.DataFrame(columns=columns) # 全結合df
    # 順に
    for e in range(n_epoch) :
        print('{}/{}'.format(e+1,n_epoch))
        for df in dfs :
            epoch_data = df[df['Epoch']==e]
            full_df = pd.concat([full_df,epoch_data])
    # CSV出力
    out_fn = 'full.csv' if out_fn is None else out_fn
    full_df.to_csv(out_fn, index=False)
    print(out_fn+' has been created')

# 別々のCSVデータとラベルを一つに結合(エポック等も変更)
def merge_csv(csv_fns, labels_fns=None, out_csv_fn=None, out_labels_fn=None) :
    # CSV結合
    for i,csv_fn in enumerate(csv_fns):
        # CSVデータ読み込み
        print('Reading '+ csv_fn)
        df = pd.read_csv(csv_fn)
        # 初期化
        if i == 0 :
            cols = df.columns # 基準のカラム
            full_df = pd.DataFrame(columns=cols) # 全結合CSV
            max_epoch = 0 # full_dfのエポック数         
        # CSVはカラムが同じでなければならない
        if cols.to_list() != df.columns.to_list() :
            raise ValueError('CSV columns must be same')
        n_epoch = np.max(df.loc[:,'Epoch'].values)+1 # dfのエポック数
        # エポックを連番に変更
        df.loc[:,'Epoch'] = df.loc[:,'Epoch'].values + max_epoch
        # 結合
        full_df = pd.concat([full_df, df])
        max_epoch += n_epoch
    # タイムを連番に変更(0から)
    sfreq = int(cols[0].split(':')[1].replace('Hz', ''))
    full_df.loc[:,cols[0]] = [i*(1/sfreq) for i in range(len(full_df.loc[:,cols[0]]))]
    # CSV出力
    out_csv_fn = 'merged.csv' if out_csv_fn is None else out_csv_fn
    full_df.to_csv(out_csv_fn, index=False)
    print(out_csv_fn + ' has been created')

    # ラベル結合
    if labels_fns is not None :
        full_labels = [] # 全結合ラベル
        for labels_fn in labels_fns :
            # ラベル読み込み
            with open(labels_fn, mode='r') as f :
                labels = [l.replace('\n', '') for l in f] # エポック毎のラベル
            print('Reading '+ labels_fn)
            # 結合
            full_labels.extend(labels)
        # ラベル出力
        out_labels_fn = 'merged_labels.txt' if out_labels_fn is None else out_labels_fn
        with open(out_labels_fn, mode='w') as f:
            f.write('\n'.join(full_labels))
        print(out_labels_fn + ' has been created')


# 特定のエポックだけ抽出したCSV作成
def extract_epochs(csv_fn, target_epochs, out_csv_fn) :
    # CSVデータ読み込み
    df = pd.read_csv(csv_fn)
    print('Reading '+csv_fn)
    out_df = pd.DataFrame(columns=df.columns)
    # 指定エポックを
    for e in target_epochs :
        out_df = pd.concat([out_df, df[df['Epoch']==e]])
    # CSV出力
    out_df.to_csv(out_csv_fn,index=False)
    print(out_csv_fn+' has been created')


# エポックのtmin~tmax秒を抽出したCSV作成
# epoch_range: 1エポックの長さ
# tmin, tmax: エポックの中で抜き出す区間
def extract_section(csv_fn, epoch_range, tmin, tmax, out_csv_fn) :
    # CSVデータ読み込み
    df = pd.read_csv(csv_fn)
    print('Reading '+csv_fn)
    out_df = pd.DataFrame(columns=df.columns)
    sfreq = int(df.columns[0].split(':')[1].replace('Hz','')) # サンプリング周波数
    n_epoch = int(np.max(df.loc[:,'Epoch'].values)) + 1 # エポック数
    # 指定範囲を
    for e in range(n_epoch) :
        start = int(sfreq * (epoch_range * e + tmin))
        end = int(sfreq * (epoch_range * e + tmax))        
        out_df = pd.concat([out_df, df.iloc[start:end]])
    # CSV出力
    out_df.to_csv(out_csv_fn,index=False)
    print(out_csv_fn+' has been created')



# Emotiv Epoc出力データを読み込み用クラス
# 変更予定(途中)
# ・Stageに対応(読込まではできる)
class EpocEEG():
    # csv_fn : EmotivEpocから出力されたCSV(列: Time:[sfreq]Hz, Epoch, CH1,...CH14, (Label)でなければならない)
    # labels_fn : 各エポックがなんのイベントに対応しているかを示すファイル(行番号=エポックで1行1ラベル)
    # include_labels : CSVがラベルのカラムを含んでいるか
    # ※EDFはエポックを読み込む方法が分からないので保留
    def __init__(self, csv_fn, labels_fn=None, include_labels=True, target_stage=None) :
        df = pd.read_csv(csv_fn)
        cols = df.columns
        # mne raw構造体を作成
        # EmotivEpocチャンネル数
        n_ch = 14
        # ch_names = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', \
        #            'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        # ch = cols.to_list()[ch_inds[0]:ch_inds[1]] # 電極数を可変にする場合
        self.n_ch = n_ch
        self.ch_names = cols.to_list()[2:2+self.n_ch] 
        data = df.loc[:, self.ch_names].values.T * 1e-6 # mne : V, epoc : μV
        self.sfreq = int(cols[0].split(':')[1].replace('Hz', '')) 
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types='eeg') # ch
        self.raw = mne.io.RawArray(data, info) # mne Raw構造体
        # self.raw.info['sfreq']
        # 
        # エポックのラベルを読み込み
        self.n_epoch = int(np.max(df.loc[:,'Epoch'].values)) + 1 # エポック数
        if labels_fn is not None :
            with open(labels_fn, mode='r') as f :
                self.epoch_labels = [label.replace('\n', '')for label in f] # エポック毎のラベル
            self.labels = sorted(set(self.epoch_labels)) # ラベル一覧
        elif include_labels:
            self.epoch_labels = [df[df['Epoch']==e].iloc[-1]['Label'] for e in range(self.n_epoch)]
            self.labels = sorted(set(self.epoch_labels))
        else :
            self.epoch_labels = [0 for e in range(self.n_epoch)]
            self.labels = ['0']
        # エポックの範囲(サンプル番号)
        self.epoch_ranges = np.zeros((self.n_epoch, 2)) # エポックの範囲(開始点, 終了点＋１)
        for e in range(self.n_epoch) :
            self.epoch_ranges[e,0] = df[df['Epoch']==e].index[0]
            self.epoch_ranges[e,1] = df[df['Epoch']==e].index[-1] 
        
        # ステージ(ない場合もある)
        if 'Stage' in cols :
            self.stages = list(np.unique(df['Stage'].values))
            self.stage_starts = dict() # ステージの開始点stage x epoch(全てのステージは連続を前提、次のステージの開始点-1が終了点)
            for stg in self.stages :
                self.stage_starts[stg] = []
                for e in range(self.n_epoch) :
                    self.stage_starts[stg].append(df[(df['Stage']==stg) & (df['Epoch']==e)].index[0])

    # データ取得(2次元)
    # target_epoch : 対象エポック(None:全範囲)
    # target_chs : 対象電極(None:全範囲)
    def get_data(self, target_epoch=None, target_chs=None):
        if target_chs is None :
            target_chs = self.ch_names
        # エポック指定
        if target_epoch is None:
            data, _ = self.raw[target_chs,:] # n_ch*n_sample
        else : 
            data, _ = self.raw[target_chs,int(self.epoch_ranges[target_epoch,0]):int(self.epoch_ranges[target_epoch,1])+1]
        return data

    '''
    データ取得
    3次元(target_epochs(labels) x target_chs x n_sample)に加工ver.
    target_epochs: 対象エポック(None: 全エポック)
    target_labels: 対象ラベル(target_epochs=Noneのとき)
    target_chs: 対象電極(None:全範囲)
    '''
    def get_split_data(self, target_epochs=None, target_labels=None, target_chs=None) :
        if target_epochs is None :
            if target_labels is None :
                target_epochs = [e for e in range(self.n_epoch)]
            else :
                target_epochs = [e for e in range(self.n_epoch) if self.epoch_labels[e] in target_labels]
        data = np.array([self.get_data(target_epoch=e, target_chs=target_chs) for e in target_epochs])
        return data

    # データ更新
    # data : 対象データ
    # target_epoch : 指定エポック(None:全範囲)
    def set_data(self, data, target_epoch=None) :
        if target_epoch is None :
            self.raw[:,:] = data
        else :
            original_data, _= self.raw[:,int(self.epoch_ranges[target_epoch,0]):int(self.epoch_ranges[target_epoch,1])+1]
            if data.shape == original_data.shape :
                self.raw[:,int(self.epoch_ranges[target_epoch,0]):int(self.epoch_ranges[target_epoch,1])+1] = data # スライスでは末尾+1


    # プロット ***n_chs != target_ch -> n_chs:何番目まで表示するか
    # target_chs : 対象チャンネル(複数可、None:全範囲)
    # target_epoch : 対象エポック(None:全範囲)
    # tmin, tmax : ・target_epoch=Noneの時
    #                0を基準に時間指定(s),
    #              ・target_epoch!=Noneの時
    #                対象エポックの開始時間を基準に時間範囲指定(s)、tmin,tmax < 0可
    #                ex. tmin=-1, tmax=5 : エポック開始時間の(-1s~+5s)の範囲(tmin < tmax, tmin:None=0, tmax:None=epoch duration)
    # scalings : グラフのy軸の大きさ
    # block : 画像を消すまで進まない
    # show : 画像を表示しない
    # out_fn : 指定した場合、画像ファイル出力()
    def plot_data(self, target_chs=None, target_epoch=None, tmin=None, tmax=None, scalings=None, block=True, show=True, title=None, out_fn=None):
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
        


    # ERP(事象関連電位)を表示
    # target_labels : 事象関連電位対象ラベル
    # target_chs : 対象電極
    # tmin, tmax : 対象ラベルのエポック開始時間を基準に時間範囲指定(s)、tmin,tmax < 0可
    #               ex. tmin=-1, tmax=5 : エポック開始時間の(-1s~+5s)の範囲(tmin < tmax, tmin:None=0,tmax:None=epoch0 duration)
    # labels_dic : ラベルとイベント番号の対応辞書(Noneで自動) 
    def plot_erp(self, target_labels, tmin=None, tmax=None, target_chs=None, labels_dic=None, title=None, show=True, out_fn=None):
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
    

    # データ保存
    # out_path : 保存ディレクトリ(無ければ作成)
    # csv_fn : データCSVファイル名
    # labels_fn : ラベルファイル(Noneなら無し)
    # include_labels : CSVにラベルのカラムを含めるか
    def save_data(self, csv_fn, labels_fn=None, include_labels=False) : 
        # 時間、エポック、電極(データ)のカラムを作成
        df = pd.DataFrame()
        time = self.raw.times
        df['Time:{}Hz'.format(int(self.sfreq))] = time
        epoch = np.zeros((len(time)))
        for e in range(self.n_epoch):
            epoch[int(self.epoch_ranges[e,0]):int(self.epoch_ranges[e,1])+1] = e
        df['Epoch'] = epoch
        ch_datas = self.get_data().T * 1e+6 
        df[self.ch_names] = ch_datas
        # ラベルを含める
        if include_labels :
            label = []
            for e in range(self.n_epoch) :
                label_range = int(self.epoch_ranges[e,1] - self.epoch_ranges[e,0]) + 1
                label.extend([self.epoch_labels[e] for i in range(label_range)])
            df['Label'] = label
        # CSV出力
        df.to_csv(csv_fn, index=False)
        print(csv_fn+' has been created')
        # ラベル出力
        if labels_fn is not None :
            with open(labels_fn, mode='w') as f:
                f.write('\n'.join(self.epoch_labels))
            print(labels_fn+' has been created')
        


class Preprocesser() :
    def __init__(self) :
        pass

    #フィルタリング
    # epoc_eeg : EpocEEGInstance
    # sfreq : サンプリング周波数
    # l_freq, h_freq : フィルタリング周波数
    #                  l_freq < h_freq: band-pass filter
    #                  l_freq > h_freq: band-stop filter
    #                  l_freq is not None and h_freq is None: high-pass filter
    #                  l_freq is None and h_freq is not None: low-pass filter
    def filter(self, epoc_eeg, l_freq, h_freq) :
        epoc_eeg.raw.filter(l_freq, h_freq)
        
    
    # 基準値減算したデータ取得(各エポック毎、各電極の平均値をオリジナルから引く)
    def remove_baseline(self, epoc_eeg) :
        new_data = np.empty((epoc_eeg.n_ch,0))
        for e in range(epoc_eeg.n_epoch) :
            epoch_data = epoc_eeg.get_data(target_epoch=e)
            epoch_means = np.mean(epoch_data, axis=1) # エポックの各チャンネル毎の平均値
            for ch in range(epoc_eeg.n_ch) :
                epoch_data[ch] = epoch_data[ch] - epoch_means[ch]    
            new_data = np.concatenate([new_data, epoch_data], axis=1)
        epoc_eeg.set_data(new_data)
        
    # ICAによるアーティファクト除去
    def remove_artifacts(self, epoc_eeg) :
        # ICA
        ica = mne.preprocessing.ICA(n_components=epoc_eeg.n_ch, n_pca_components=epoc_eeg.n_ch, max_iter=100)
        ica.fit(epoc_eeg.raw)
        # アーティファクト除去
        ica.detect_artifacts(epoc_eeg.raw)
        ica.apply(epoc_eeg.raw)
        


