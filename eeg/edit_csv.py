# EmotivEpoc出力のCSVファイル操作関数
# KARONEのやつもいれてしまおう
import numpy as np
import pandas as pd


'''
CSVにラベルを追加
csv_fn: csv名
labels_fn: ラベルファイル名(各行がエポックに対応) 
out_fn: 出力ファイル名(Noneならcsv_fnに上書き)
'''
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

'''
CSVファイルを結合
csv_fns: CSV名リスト(順番)
out_fn: 出力ファイル名
'''
def concat_csv(csv_fns, out_fn) :
    # CSV結合
    for i,csv_fn in enumerate(csv_fns):
        # CSVデータ読み込み
        print('Reading '+ csv_fn)
        df = pd.read_csv(csv_fn)
        # 初期化
        if i == 0 :
            cols = df.columns # 基準のカラム
            concat_df = pd.DataFrame(columns=cols) # 全結合CSV
            max_epoch = 0 # concat_dfのエポック数   

        # CSVはカラムが同じでなければならない
        if cols.to_list() != df.columns.to_list() :
            raise ValueError('CSV columns must be same')

        n_epoch = np.max(df.loc[:,'Epoch'].values)+1 # dfのエポック数

        # dfのエポックはconcat_dfの続きからになる
        df.loc[:,'Epoch'] = df.loc[:,'Epoch'].values + max_epoch

        # 結合
        concat_df = pd.concat([concat_df, df])
        max_epoch += n_epoch

    # タイムを0からに変更(0から)
    sfreq = int(cols[0].split(':')[1].replace('Hz', ''))
    concat_df.loc[:,cols[0]] = [i*(1/sfreq) for i in range(len(concat_df.loc[:,cols[0]]))]
    
    # CSV出力
    concat_df.to_csv(out_fn, index=False)
    print(out_fn + ' has been created')

    
'''
ステージ単位で分けられたCSVファイルを時系列順に並び替えて結合
ex. rest.csv(rest1, rest2...), thinking.csv(thinking1, thinking2...) 
    -> full.csv(rest1, thinking1, rest2...)
csv_fns: csvファイルリスト(ステージの順通りに)
out_fn: 出力ファイル名(Noneならcsv_fnに上書き)
'''
def sort_csv(csv_fns, out_fn=None):
    dfs = [] # CSVs
    for csv_fn in csv_fns :
        print('Reading '+csv_fn)
        dfs.append(pd.read_csv(csv_fn))
    
    # 1つ目のCSVを基準(エポック数、列名)
    n_epoch = np.max(dfs[0].loc[:,'Epoch'].values)+1
    columns = dfs[0].columns
    full_df = pd.DataFrame(columns=columns) # 全結合df
    
    # 時系列に直して結合
    for e in range(n_epoch) :
        for df in dfs :
            epoch_data = df[df['Epoch']==e]
            full_df = pd.concat([full_df,epoch_data])
    # CSV出力
    out_fn = 'full.csv' if out_fn is None else out_fn
    full_df.to_csv(out_fn, index=False)
    print(out_fn+' has been created')

'''
特定のエポックだけ抽出したCSV作成
csv_fn: CSV名
target_epochs: 対象エポックリスト
out_fn: 出力ファイル名
'''
def extract_epochs(csv_fn, target_epochs, out_fn) :
    # CSVデータ読み込み
    df = pd.read_csv(csv_fn)
    print('Reading '+csv_fn)
    out_df = pd.DataFrame(columns=df.columns)
    # 指定エポックを
    for e in target_epochs :
        out_df = pd.concat([out_df, df[df['Epoch']==e]])
    # CSV出力
    out_df.to_csv(out_fn,index=False)
    print(out_fn+' has been created')


'''
エポックのtmin~tmax秒を抽出したCSV作成
tmin, tmax: エポックの中で抜き出す区間(sec)
out_fn: 出力ファイル名
'''
def extract_section(csv_fn, tmin, tmax, out_fn) :
    # CSVデータ読み込み
    df = pd.read_csv(csv_fn)
    print('Reading '+csv_fn)
    out_df = pd.DataFrame(columns=df.columns)
    sfreq = int(df.columns[0].split(':')[1].replace('Hz','')) # サンプリング周波数
    n_epoch = int(np.max(df.loc[:,'Epoch'].values)) + 1 # エポック数
    # 指定範囲を
    for e in range(n_epoch) :
        start = int(sfreq * tmin)
        end = int(sfreq *tmax) 
        out_df = pd.concat([out_df, df[df['Epoch']==e].iloc[start:end]]) 
    # CSV出力
    out_df.to_csv(out_fn, index=False)
    print(out_fn+' has been created')