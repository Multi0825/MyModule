# 自作汎用関数群 
import os
import time
import datetime
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

# ルートディレクトリ取得
def get_root_dir() :
    cwd = os.getcwd()
    root = cwd.split('\\')[0]+'/'
    return root


# ファイル出力のプリントを楽に
# txt : 出力する文字列
# out_fn : 出力ファイル名
# reset : ファイルを空にするか
def print_ex(txt, out_fn=None, reset=False) :
    if out_fn==None :
        print(txt)
    else :
        if reset :
            mode = 'w'
        else :
            mode = 'a'
        out_f = open(out_fn, mode=mode)
        print(txt, file=out_f)
        out_f.close()

# 時間取得
class timer() :
    def __init__(self) :
        self.start = time.time() # 開始
        self.elapsed = 0 # 経過
    # 開始
    def start(self) :
        self.start = time.time()
    # 経過
    def elapsed(self, HMS=False, MS=False) :
        self.elapsed = time.time() - self.start
        return self.elapsed
        
    @staticmethod
    def sec2min(sec) :
        return sec//60
    @staticmethod
    def sec2hour(sec) :
        return sec//3600
    @staticmethod
    def sec2hm(sec) :
        return sec//3600, sec%3600//60
    @staticmethod
    def sec2hms(sec) :
        return sec//3600, sec%3600//60, sec%3600%60
    
    
# pklファイル保存
def pkl_save(variable, pkl_fn) :
    with open(pkl_fn, mode='wb') as f:
        pkl.dump(variable, f)
    print(pkl_fn + ' has been created')

# pklファイル読み込み
def pkl_load(pkl_fn) :
    with open(pkl_fn, mode='rb') as f:
        variable = pkl.load(f)
    print(pkl_fn + ' has been loaded')
    return variable

# パスファイル(オリジナル)を読込
# パスファイル定義
# 1行に1パス、PathId=path
# *+Id=path(ex. *1=/dir)で書かれたものは別のパスに使える(PathId1=*1/)
# 特殊なファイル名による不具合は知らない
def read_path(path_fn) :
    path_dict = {}
    commons = {}
    with open(path_fn, mode='r') as f :
        for l in f :
            l = l.replace('\n', '').split('=')
            if len(l) == 2 : # 空行対策
                path_id = l[0]
                path = l[1]
                # "*+Num"を置換(その行までに出てきているもの)
                for c in commons.items() :
                    if c[0] in path :
                        path = path.replace(c[0],c[1])
                # IDが"*+Num"を含むときは記録
                if '*'in path_id :
                    commons[path_id] = path
                path_dict[path_id] = path
    return path_dict

################################## 未完 ##############################################
# グラフパラメータ
class FigParam() :
    def __init__(self, x, y, label='', title='', color=None, xlabel=None, ylabel=None, xlim=None, ylim=None, plot_type='plot') :
        self.x = x
        self.y = y
        self.label = label
        self.title = title 
        self.color = color
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.plot_type = plot_type

# グラフ
class Figure() :
    def __init__(self,figsize=[6.4, 4.8]) :
        self.fig = plt.figure(figsize=figsize)
        self.fig_params = []
    
    # 追加
    # type: plot, scatter, bar 
    def add_ax(self, x, y, label=None, title='', color=None, xlabel=None, ylabel=None, xlim=None, ylim=None, plot_type='plot' ) :
        self.fig_params.append(FigParam(x, y, label, title, color, xlabel, ylabel, xlim, ylim, plot_type))

    # rc_params : 図のパラメータを設定
    # ex. 
    # 'figure.size' :　図のの大きさ(inch)
    # 'font.size' : フォントの大きさ
    # 'axes.grid' : グリッドを表示するか(bool)
    def set_rcparams(self, rc_params) :
        for param in rc_params.items() :
            plt.rcParams[param[0]] = param[1]

    # プロット
    def plot(self, fig_shape=None, one_fig=False) :
        n_fig = len(self.fig_params)# 画像数
        if n_fig == 0 :
            raise ValueError('Num of fig is 0, First add_ax')
        n_fig = 1 if one_fig else n_fig 
        
        # 図形状
        if fig_shape == None :
            fig_shape = (n_fig, 1)
        print(n_fig)
        print(fig_shape)
        if n_fig == 1 :
            ax = self.fig.add_subplot(fig_shape[0], fig_shape[1], 1, title=self.fig_params[0].title)
            if self.fig_params[1].label != None :
                ax.legend()
            for i in range(n_fig) :
                ax.plot(self.fig_params[i].x, self.fig_params[i].y, label=self.fig_params[i].label)
                # 本当はあとsetxlimだなんだとあるが面倒なので後で
        else :
            for i in range(n_fig) :
                ax = self.fig.add_subplot(fig_shape[0], fig_shape[1], i+1, title=self.fig_params[i].title) # row, column, index 1 2
                                                                                                         #                    3 4
                ax.plot(self.fig_params[i].x, self.fig_params[i].y, label=self.fig_params[i].label)
                if self.fig_params[i].label != None :
                    ax.legend()
        self.fig.tight_layout()
        plt.show()

    # 保存
    def save(self, out_fn) :
        self.fig.savefig(out_fn)
