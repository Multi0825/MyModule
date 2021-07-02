import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# 主成分分析(PCA)の結果をCSVに出力
# tr_data: 訓練データ(n_samples * n_feats)
# test_data: テストデータ(指定しない場合、訓練をそのまま)
# labels: 結果用データの各サンプルのラベル 
# feat_labels: 特徴のラベル
# n_components: 成分数(None: min(n_samples, n_feats)-1)
# comps_plt: 描画する成分番号[X,Y]
# XXX_fn: 各種ファイル名
def pca(tr_data, test_data=None, labels=None, feat_labels=None, 
        n_components=None, comps_plt=[1,2],
        feat_fn='./pca_feat.csv', plt_fn='./pca.png', evr_fn='./evr.csv', eigen_fn='./eigen.csv') :
    pca = PCA(n_components=n_components)
    if test_data is None :
        feats = pca.fit_transform(tr_data)
    else :
        pca.fit(tr_data)
        feats = pca.transform(test_data)
    
    # 得点(X:主成分, Y:サンプル)
    index = ['Sample{}'.format(i+1) for i in range(feats.shape[0])] if labels is None else labels
    feat_df = pd.DataFrame(feats, index=index, 
                           columns=["PC{}".format(i+1)for i in range(feats.shape[1])])
    feat_df.to_csv(feat_fn)
    print('PCA Feature Results: {}'.format(feat_fn))

    # グラフ(X, Y:成分)
    plt.figure()
    if labels is None :
        plt.scatter(feat_df.iloc[:,comps_plt[0]],feat_df.iloc[:,comps_plt[1]],alpha=0.8)
    else :
        for sl in set(labels) :
            plt.scatter(feat_df.loc[sl,'PC'+str(comps_plt[0])],feat_df.loc[sl,'PC'+str(comps_plt[1])],alpha=0.8, label=sl)
    plt.grid()
    plt.legend()
    plt.xlabel("PC{}".format(comps_plt[0]))
    plt.ylabel("PC{}".format(comps_plt[1]))
    plt.savefig(plt_fn)
    print('Plot PCA Feature(X:{}, Y:{}): {}'.format(comps_plt[0], comps_plt[1],plt_fn))

    # 寄与率(第何主成分まででどのくらいの情報を説明できるか)
    evr_df = pd.DataFrame(pca.explained_variance_ratio_, index=['PC{}'.format(i+1) for i in range(feats.shape[1])])
    evr_df.to_csv(evr_fn, header=False)
    print('Explained Variance Ratio Results: {}'.format(evr_fn))

    # 固有ベクトル(成分の軸に対し、どの特徴による変化が大きいかを示す)(X:特徴, Y:主成分)
    eigen_df = pd.DataFrame(pca.components_, columns=feat_labels, index=['PC{}'.format(i+1) for i in range(feats.shape[1])])
    eigen_df.to_csv(eigen_fn, header=(feat_labels is not None))
    print('Eigen Vectors Results: {}'.format(eigen_fn))

# グラフ(X, Y:成分)だけ作成(共通だからやり直すのが無駄)
def plot_pc(pca_feats_fn, comps_plt=[1,2], plt_fn='./pca.png') :
    feat_df = pd.read_csv(pca_feats_fn)
    labels = feat_df.index
    if len(set(labels)) == len(labels) :
        labels = None 
    # グラフ(X, Y:成分)
    plt.figure()
    if labels is None :
        plt.scatter(feat_df.iloc[:,comps_plt[0]],feat_df.iloc[:,comps_plt[1]],alpha=0.8)
    else :
        for sl in set(labels) :
            plt.scatter(feat_df.loc[sl,'PC'+str(comps_plt[0])],feat_df.loc[sl,'PC'+str(comps_plt[1])],alpha=0.8, label=sl)
    plt.grid()
    plt.legend()
    plt.xlabel("PC{}".format(comps_plt[0]))
    plt.ylabel("PC{}".format(comps_plt[1]))
    plt.savefig(plt_fn)
    print('Plot PCA Feature(X:{}, Y:{}): {}'.format(comps_plt[0], comps_plt[1],plt_fn))