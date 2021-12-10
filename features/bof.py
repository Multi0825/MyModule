# Bag of Features
import numpy as np
import cv2

class BOF() :
    # n_clusters: クラスター数
    # feats: 特徴量(n_samples x n_vecs x n_feats)
    def __init__(self, n_clusters, feats) :
        feats = feats.astype(np.float32) # f32に
        self.n_clusters = n_clusters
        self.bof_trainer = cv2.BOWKMeansTrainer(n_clusters)
        for f in feats :
            self.bof_trainer.add(f)
        self.codebook = self.bof_trainer.cluster() # n_clusters x n_feats

    # ユークリッド距離が近いcodewordに変換
    # feature: n_vecs x n_feats
    def replace_codeword(self, feat_vecs) :
        codewords = np.zeros((1,self.codebook.shape[1]))
        for fv in feat_vecs :
            dists = np.linalg.norm(self.codebook - fv, axis=1) # ユークリッド距離
            min_ind = np.argmin(dists)
            codewords = np.concatenate([codewords, self.codebook[min_ind].reshape(1,-1)])
        codewords = codewords[1:]
        return codewords
    
    # ユークリッド距離が近いcodewordに変換、ヒストグラムで表現
    # feature: n_vecs x n_feats
    def histogram(self, feat_vecs) :
        min_inds = []
        for fv in feat_vecs :
            dists = np.linalg.norm(self.codebook - fv, axis=1) # ユークリッド距離
            min_ind = np.argmin(dists)
            min_inds.append(min_ind)
        hist = [min_inds.count(i) for i in range(len(self.codebook))]
        return hist
