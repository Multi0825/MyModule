# 自作汎用関数群
# 2021/07/10 sakai_utilも統合(面倒だったため)
import os
import time
import pickle as pkl
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix


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

'''
パスファイル(オリジナル)を読込
パスファイル定義
1行に1つ、Id=Path, or *Id=Path
*Idは別のパスに配置で置換
ex. *id1=/dir1
    id2=*id1/dir2/
    id3=*id1/dir3/
先頭に#で無視
＊特殊なディレクトリ名による不具合は知らない
'''
def read_path(path_fn) :
    path_dict = {} # Id:path
    ast_path = {} # *Idを記録
    with open(path_fn, mode='r') as f :
        for l in f :
            l = l.replace('\n', '').split('=')
            if len(l) == 2 : # 空行対策
                path_id = l[0]
                path = l[1]
                if '#' is not path_id[0] :
                    # *Idがあれば対応するパスに置換(その行までに出てきているもの)
                    for c in ast_path.items() :
                        if c[0] in path :
                            path = path.replace(c[0],c[1])
                    # *Idを記録
                    if '*'in path_id :
                        ast_path[path_id] = path
                    path_dict[path_id] = path
    return path_dict

'''
read_path()で作成したpath_dictに掲載されたディレクトリをすべて作成
ignore_ast: *Idを無視
'''
def make_path(path_dict, ignore_ast=True) -> None :
    for item in path_dict.items() :
        if ignore_ast and ('*' in item[0]) :
            pass
        else :
            os.makedirs(item[1], exist_ok=True)


# 重複のない乱数生成
# a, b: a <= x < b (random.randintと異なる)
# n: 個数
def no_duplicated_randint(a, b, n, seed=None):
    if n > np.abs(a-b) :
        raise ValueError('Must be n>=|a-b|')
    x = [i for i in range(a, b)]
    y = []
    for i in range(n) :
        j = random.randint(0, len(x)-1)
        y.append(x.pop(j))
    return y

# データ量を均一化(少ない方を増加)
def equalize(a, b, random_seed=None) :
    a = list(a)
    b = list(b)
    add_inds = []
    a_is_more = True if len(a) > len(b) else False # 戻り値の順番保持用
    more = a if a_is_more else b
    less = b if a_is_more else a
    n_more = len(more)
    n_less = len(less)
    # できる限り重複を減らす
    while n_more > n_less :
        # 2倍以上差があるときはn_lessだけ追加
        if n_more >= 2*n_less:
            add_inds.extend(no_duplicated_randint(0, n_less, n_less))
            n_less = 2 * n_less
        # さもなくば差分追加
        else :
            add_inds.extend(no_duplicated_randint(0, n_less, n_more-n_less, seed=random_seed))
            n_more = n_less
    # 追加
    for ai in add_inds :
        less.append(less[ai])

    # 入力の順を保持
    if a_is_more :
        return more, less # 戻り値はリスト化される
    else :
        return less, more

# データの順番をシャッフル
def shuffle(data, random_seed=None) :
    random.seed(random_seed)
    random.shuffle(data)
    return data

# データの標準化(mean=0, var=1)
# data: dim=2
# axis: 
def standardize(data, axis=None):
    return stats.zscore(data,axis=axis)

# データ正規化(min:0, max:1)
# data: n_ch * n_sample
def normalize(data, axis=0) :
    mms = preprocessing.MinMaxScaler() 
    # fit_transform: データ数*特徴量を特徴量毎(縦)に正規化
    if axis==1 :
        nor_data = mms.fit_transform(data.T).T 
    elif axis==0: 
        nor_data = mms.fit_transform(data)
    else :
        raise ValueError('axis = 0 or 1')
    return nor_data

# 分類結果評価
# y_pred: 予測
# y: 正解
def eval_classification(y_pred, y) :
    ac_score  = accuracy_score(y, y_pred) # 正解率
    precision = precision_score(y, y_pred) # 適合率
    recall    = recall_score(y, y_pred) # 再現率
    f1        = f1_score(y, y_pred) # F1値
    kappa     = cohen_kappa_score(y, y_pred) # κ係数
    conf      = confusion_matrix(y, y_pred) # 混同行列
    return ac_score, precision, recall, f1, kappa, conf

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
