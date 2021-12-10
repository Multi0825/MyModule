# 脳波から画像を生成
# 1. 電極座標を3Dから2Dに
# 2. 各電極の配置と各電極の特徴量から任意のサイズの画像(行列)を生成(n_sample x n_feat x L x L)
# 参考
# LEARNING REPRESENTATIONS FROM EEG WITH DEEP RECURRENT-CONVOLUTIONAL NEURAL NETWORKS
# https://github.com/pbashivan/EEGLearn/blob/master/eeglearn/eeg_cnn_lib.py
# https://github.com/pbashivan/EEGLearn/blob/master/eeglearn/utils.py
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
from sklearn.preprocessing import scale

##### ほぼ参考まま #####
def cart2sph(x, y, z):
    """
    Transform Cartesian coordinates to spherical
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return: radius, elevation, azimuth
    """
    x2_y2 = x**2 + y**2
    r = math.sqrt(x2_y2 + z**2)                    # r
    elev = math.atan2(z, math.sqrt(x2_y2))            # Elevation
    az = math.atan2(y, x)                          # Azimuth
    return r, elev, az

def pol2cart(theta, rho):
    """
    Transform polar coordinates to Cartesian
    :param theta: angle value
    :param rho: radius value
    :return: X, Y
    """
    return rho * math.cos(theta), rho * math.sin(theta)

def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.
    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, math.pi / 2 - elev)


def augment_EEG(data, stdMult, pca=False, n_components=2):
    """
    Augment data by adding normal noise to each feature.
    :param data: EEG feature data as a matrix (n_samples x n_features)
    :param stdMult: Multiplier for std of added noise
    :param pca: if True will perform PCA on data and add noise proportional to PCA components.
    :param n_components: Number of components to consider when using PCA.
    :return: Augmented data as a matrix (n_samples x n_features)
    """
    augData = np.zeros(data.shape)
    if pca:
        pca = PCA(n_components=n_components)
        pca.fit(data)
        components = pca.components_
        variances = pca.explained_variance_ratio_
        coeffs = np.random.normal(scale=stdMult, size=pca.n_components) * variances
        for s, sample in enumerate(data):
            augData[s, :] = sample + (components * coeffs.reshape((n_components, -1))).sum(axis=0)
    else:
        # Add Gaussian noise with std determined by weighted std of each feature
        for f, feat in enumerate(data.transpose()):
            augData[:, f] = feat + np.random.normal(scale=stdMult*np.std(feat), size=feat.size)
    return augData

# 画像生成
def gen_images(locs, features, n_gridpoints, normalize=True,
               augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode
    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param std_mult     Multiplier for std of added noise
    :param n_components: Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    feat_array_temp = []
    nElectrodes = locs.shape[0]     # Number of electrodes
    n_samples = features.shape[0]

    # Test whether the feature vector length is divisible by number of electrodes
    # 色の数
    assert features.shape[1] % nElectrodes == 0
    n_colors = features.shape[1] // nElectrodes
    
    # α, β等帯域分け
    for c in range(n_colors):
        feat_array_temp.append(features[:, nElectrodes * c : nElectrodes * (c+1)])
    
    # 略
    if augment: # default False
        if pca: # default False
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=True, n_components=n_components)
        else:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=False, n_components=n_components)    

    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                     ] # mgrid(min_x:max_x:step, min_y:max_y:step): minからmaxまでstepずつ格子状の配列(なぜ虚数かは不明)
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([n_samples, n_gridpoints, n_gridpoints]))

    # Generate edgeless images
    if edgeless: # default False
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((n_samples, 4)), axis=1)

    # Interpolating
    for i in range(n_samples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                               method='cubic', fill_value=np.nan)
        # print('Interpolating {0}/{1}\r'.format(i + 1, n_samples), end='\r')

    # Normalizing
    for c in range(n_colors):
        if normalize: # default True
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])
    return np.swapaxes(np.asarray(temp_interp), 0, 1)     # swap axes to have [samples, colors, W, H]


# 手順を一括
# ch_cords3d: np.ndarray, n_ch x 3(x, y, z)
# feats: np.ndarray, n_sample x n_ch x n_feat
# n_gridpoints: 画像(変換行列)
# others: gen_images参照
def gen_eeg_imgs(ch_cords3d, feats, n_gridpoints, 
                 normalize=True, augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False) :
    n_ch = ch_cords3d.shape[0]
    ch_cords2d = []
    # 3D -> 2D
    for ch in range(n_ch) :
        cord3d = [ch_cords3d[ch, 0], ch_cords3d[ch, 1], ch_cords3d[ch, 2]]
        x, y = azim_proj(cord3d)
        ch_cords2d.append([x, y])
    ch_cords2d = np.array(ch_cords2d)
    # 画像生成
    # feats並び替え n_sample x n_ch x n_feat -> n_sample x (a_1,...,a_n_ch,b1,...,b_n_ch,...)
    feats_new = np.array([feats[i].T.reshape(feats.shape[1]*feats.shape[2]) for i in range(feats.shape[0])])
    imgs = gen_images(locs=ch_cords2d, features=feats_new, n_gridpoints=n_gridpoints, 
                      normalize=normalize, augment=augment, pca=pca, 
                      std_mult=std_mult, n_components=n_components, edgeless=edgeless)
    return imgs # n_sample x D(画像分野におけるチャンネル) x W x H 

# 画像を表示or出力死体場合
# img: D x W x H
# cmap: カラーマップ
# fig_fn: 出力ファイル名
def plt_img(img, cmap='jet', fig_fns=None) :
     for d in range(img.shape[0]) :
        x = [i for i in range(img.shape[1])]
        y = [i for i in range(img.shape[2])]
        z = img[d].T # 転置しないと軸が非直観的 
        plt.contourf(x, y, z, cmap=cmap)
        plt.colorbar()
        if fig_fns is None :
            plt.show()
            plt.close()
        elif len(fig_fns) == img.shape[0] :
            plt.savefig(fig_fns[d])
            plt.close()
        else :
            raise ValueError('fig_fns=[fig_fn1, fig_fn2,..., fig_fnN] N=img.shape[0]')
            

