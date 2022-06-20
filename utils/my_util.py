# 自作汎用関数群
import os
import time
import pickle as pkl
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import preprocessing
import math
from decimal import Decimal, ROUND_HALF_UP
from logging import getLogger, Formatter, StreamHandler, FileHandler
import itertools

def get_root_dir() :
    '''
    ルートディレクトリ取得
    '''
    cwd = os.getcwd()
    root = cwd.split('\\')[0]+'/'
    return root

def print_ex(txt, out_fn=None, reset=False) :
    '''
    ファイル出力のプリントを楽に(Loggerで良かった)
    txt : 出力する文字列
    out_fn : 出力ファイル名
    reset : ファイルを空にするか
    '''
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

def read_path(path_fn) :
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


def make_path(path_dict, ignore_ast=True) -> None :
    '''
    read_path()で作成したpath_dictに掲載されたディレクトリをすべて作成
    ignore_ast: *Idを無視
    '''
    for item in path_dict.items() :
        if ignore_ast and ('*' in item[0]) :
            pass
        else :
            os.makedirs(item[1], exist_ok=True)


def no_duplicated_randint(a, b, n, random_seed=None):
    '''
    重複のない乱数生成
    a, b: a <= x < b (random.randintと異なる)
    n: 個数
    '''
    if n > np.abs(a-b) :
        raise ValueError('Must be n>=|a-b|')
    x = [i for i in range(a, b)]
    y = []
    for i in range(n) :
        random.seed(random_seed)
        j = random.randint(0, len(x)-1)
        y.append(x.pop(j))
    return y

def equalize(a, b, increase=False,random_seed=None) :
    '''
    データ量を均一化(少量増加or多量減少)
    a,b: data
    increase: True=少量増加, False=多量減少
    random_seed: seed
    '''
    # 増加
    if increase :
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
                add_inds.extend(no_duplicated_randint(0, n_less, n_less, random_seed=random_seed))
                n_less = 2 * n_less
            # さもなくば差分追加
            else :
                add_inds.extend(no_duplicated_randint(0, n_less, n_more-n_less, random_seed=random_seed))
                n_more = n_less
        # 追加
        for ai in add_inds :
            less.append(less[ai])

        # 入力の順を保持
        if a_is_more :
            return more, less # 戻り値はリスト化される
        else :
            return less, more
    # 減少
    else :
        a = list(a)
        b = list(b)
        a_is_more = True if len(a) > len(b) else False # 戻り値の順番保持用
        more = a if a_is_more else b
        less = b if a_is_more else a
        n_more = len(more)
        n_less = len(less)

        # 多い方からランダムに削除
        n_remained = no_duplicated_randint(0, n_more, n_less, random_seed=random_seed)
        more = [more[n] for n in n_remained]

        # 入力の順を保持
        if a_is_more :
            return more, less # 戻り値はリスト化される
        else :
            return less, more

def shuffle(data, random_seed=None) :
    '''
    データの順番をシャッフル
    '''
    random.seed(random_seed)
    random.shuffle(data)
    return data


def standardize(data, axis=None):
    '''
    データの標準化(mean=0, var=1)
    data: dim=2
    axis: 
    '''
    return stats.zscore(data,axis=axis)

# 
def normalize(data, axis=0) :
    '''
    データ正規化(min:0, max:1)
    data: n_ch * n_sample
    axis: 
    '''
    mms = preprocessing.MinMaxScaler() 
    # fit_transform: データ数*特徴量を特徴量毎(縦)に正規化
    if axis==1 :
        nor_data = mms.fit_transform(data.T).T 
    elif axis==0: 
        nor_data = mms.fit_transform(data)
    else :
        raise ValueError('axis = 0 or 1')
    return nor_data

def rnd(n, digit=0.1) :
    '''
    指定桁を四捨五入
    digit: 桁指定(..., 10, 1, 0.1,...)
    * numpy.ndarrayとかでエラー
    '''
    digit = digit*10 # デフォルトが非直観的だったから修正(0指定すると0.1を四捨五入みたいな)
    digit = '1E{}'.format(int(math.log10(digit))) if digit>=1 else str(digit)
    return Decimal(n).quantize(Decimal(digit), rounding=ROUND_HALF_UP)

class Logger() :
    '''
    Logger用クラス
    '''
    def __init__(self, logger_level=1, stream=True, stream_level=1) :
        '''
        logger_level: 基本レベル(基本レベル以下は無視)
        stream: 標準出力有無
        stream_level: 標準出力レベル(レベル以上のlogのみ表示)
        '''
        self.logger = getLogger('Logger')
        self.logger_level = logger_level
        self.logger.setLevel(self.logger_level)
        self.handlers = {}
        if stream :
            self.handlers['stream'] = StreamHandler()
            self.handlers['stream'].setLevel(stream_level)
            self.logger.addHandler(self.handlers['stream'])

    def log(self, msg, level=1) :
        '''
        ログ表示
        msg: 本文
        level: 出力レベル(レベル未満のHandlerには表示されない)
        '''
        self.logger.log(level, msg)

    
    def add_handler(self, handler_id, file_name, mode='w', level=1, stream=False) :
        '''
        FileHandler(Stream Handler)追加
        handler_id: id
        file_name: ファイル名
        mode: 'w', 'a'
        level: 出力レベル(レベル以上のlogのみ表示)
        stream: Trueの場合、StreamHandlerに(file_name, modeは無効)
        '''
        if handler_id in self.handlers.keys() :
            self.remove_handler(handler_id)
        self.handlers[handler_id] = FileHandler(file_name, mode) if not stream else StreamHandler()
        self.handlers[handler_id].setLevel(level)
        self.logger.addHandler(self.handlers[handler_id])
            
    def remove_handler(self, handler_id) :
        '''
        Handler削除
        handler_id: id
        '''
        self.logger.removeHandler(self.handlers[handler_id])
    
    def set_level(self, handler_id, level) :
        '''
        Handlerレベル変更
        handler_id: id
        level: 変更後
        '''
        self.handlers[handler_id].setLevel(level)

    def set_format(self, handler_id, fmt) :
        '''
        フォーマット追加
        handler_id: id
        fmt: 特殊文字列(無効にしたい場合、'')
        一覧(https://srbrnote.work/archives/4472)
        %(asctime)s         -> 2019-11-03 13:59:56,644 -> 時刻 (人が読める形式)
        %(created)f         -> 1572757196.645207 -> 時刻 (time.time()形式)
        %(filename)s        -> test_log_formats.py -> ファイル名
        %(funcName)s        -> test_formats -> 関数名
        %(levelname)s       -> DEBUG -> ロギングレベルの名前
        %(levelno)s         -> 10 -> ロギングレベルの数値
        %(lineno)d          -> 81 -> 行番号
        %(module)s          -> test_log_formats -> モジュールの名前
        %(msecs)d           -> 646 -> 時刻のミリ秒部分 (milliseconds)
        %(name)s            -> __main__ -> ロガーの名前
        %(pathname)s        -> F:\apps\data\test_log_formats.py -> ファイルパス
        %(process)d         -> 8676 -> プロセスID (PID)
        %(processName)s     -> MainProcess -> プロセス名
        %(relativeCreated)d -> 5 -> logging モジュール読み込みからの時刻 (ミリ秒)
        %(thread)d          -> 4788 -> スレッドID
        %(threadName)s      -> MainThread -> スレッド名
        '''
        self.handlers[handler_id].setFormatter(Formatter(fmt))

def gen_paramset(param_fn, paramset_fn='paramset.csv'):
    '''
    param.txt
        param1:val1,val2,...
        param2:val1,val2,...
        ...
    から組み合わせ全通りのリストを生成
    paramset.csv
        id_set,param1,param2,...
        0,val1,val1,...
        1,val1,val2,...
    '''
    # パラメータ読み込み
    with open(param_fn, 'r') as rf, open(paramset_fn, 'w') as wf :
        paramset = {} # param:[val1,val2,...]
        for l in rf :
            param, vals = l.replace('\n', '').split(':')
            paramset[param] = vals.split(',')
        n_param = len(paramset.keys())
        vals = list(paramset.values()) 
        val_combs = list(itertools.product(*vals)) 
        wf.write(('{}'+',{}'*n_param+'\n').format('id_set',*paramset.keys()))
        
        for id_set, vc in enumerate(val_combs) :
            wf.write(('{}'+',{}'*n_param+'\n').format(id_set,*vc))

def iter_paramset(paramset_fn) :
    '''
    gen_paramsetで生成したCSVを元にイテレーション
    '''
    with open(paramset_fn,'r') as f :
        params = f.readline().replace('\n','').split(',')
        val_combs = []
        for l in f :
            val_comb = {}
            for p,v in zip(params,l.replace('\n','').split(',')) :
                # pdb.set_trace()
                try: # float or int
                    val_comb[p] = float(v) if '.' in v else int(v)
                except ValueError: # str
                    val_comb[p] = v
            val_combs.append(val_comb)
    for vc in val_combs :
        yield vc # dict({id_set:val, param1:val,...})
