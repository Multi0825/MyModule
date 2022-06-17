import glob
from oct2py import octave
import oct2py

def validate_eeglab(eeglab_dir, 
                    rem_funcs=('@mmo','@memmapdata','@eegobj'), 
                    plugins=('firfilt',),
                    signal_dir='/usr/share/octave/packages/signal-1.4.2/'):
    '''
    eeglabを有効化
    eeglab_dir: eeglabのディレクトリ
    rem_funcs: eeglab/functionsの中の不要なディレクトリ名、エラーが出たりするものはここに
    plugins: plugin下で必要なもの
    signal_dir: eeglabに必要なsignal packageのパス(一応可変にしておく)
    '''
    # eeglab functionsパス
    funcs_dir = eeglab_dir + '/functions/'
    funcs = glob.glob(funcs_dir+'*')
    for rf in rem_funcs :
        funcs.remove(funcs_dir+rf)
    for f in funcs :
        octave.addpath(f)
    # eeglab pluginsパス
    plugins_dir = eeglab_dir + '/plugins/'
    for p in plugins :
        octave.addpath(plugins_dir+p)
    # signal packageパス(eeglabに必要)
    octave.addpath(signal_dir)

class StructEEGLAB(oct2py.io.Struct) :
    '''
    OctaveでEEGLABを動かすときの構造体
    oct2py.io.Structからのダウンキャストにも対応したい
    '''
    def __init__(self, 
                 data, times, 
                 nbchan, srate, 
                 **kwargs) :
        '''
        必須であろう変数が引数
        その他は省略
        '''
        super().__init__()
        # 指定が無ければ基本空のリスト(計算ができ、かつ指定無でエラーが出るものは値)
        self['data'] = data
        self['times'] = times
        self['srate'] = srate
        self['nbchan'] = nbchan
        self['pnts'] = kwargs['pnts'] if 'pnts' in kwargs else len(times)
        self['xmin'] = kwargs['xmin'] if 'xmin' in kwargs else times[0]
        self['xmax'] = kwargs['xmax'] if 'xmax' in kwargs else times[-1]
        self['setname'] = kwargs['setname'] if 'setname' in kwargs else []
        self['filename'] = kwargs['filename'] if 'filename' in kwargs else []
        self['filepath'] = kwargs['filepath'] if 'filepath' in kwargs else []
        self['subject'] = kwargs['subject'] if 'subject' in kwargs else []
        self['group'] = kwargs['group'] if 'group' in kwargs else []
        self['condition'] = kwargs['condition'] if 'condition' in kwargs else []
        self['session'] = kwargs['session'] if 'session' in kwargs else []
        self['comments'] = kwargs['comments'] if 'comments' in kwargs else []
        self['trials'] = kwargs['trials'] if 'trials' in kwargs else 1.0
        self['icaact'] = kwargs['icaact'] if 'icaact' in kwargs else []
        self['icawinv'] = kwargs['icawinv'] if 'icawinv' in kwargs else []
        self['icasphere'] = kwargs['icasphere'] if 'icasphere' in kwargs else []
        self['icaweights'] = kwargs['icaweights'] if 'icaweights' in kwargs else []
        self['icachansind'] = kwargs['icachansind'] if 'icachansind' in kwargs else []
        self['chanlocs'] = kwargs['chanlocs'] if 'chanlocs' in kwargs else []
        self['urchanlocs'] = kwargs['urchanlocs'] if 'urchanlocs' in kwargs else []
        self['chaninfo'] = kwargs['chaninfo'] if 'chaninfo' in kwargs else []
        self['ref'] = kwargs['ref'] if 'ref' in kwargs else []
        self['event'] = kwargs['event'] if 'event' in kwargs else []
        self['urevent'] = kwargs['urevent'] if 'urevent' in kwargs else []
        self['eventdescription'] = kwargs['eventdescription'] if 'eventdescription' in kwargs else []
        self['epoch'] = kwargs['epoch'] if 'epoch' in kwargs else []
        self['epochdescription'] = kwargs['epochdescription'] if 'epochdescription' in kwargs else []
        self['reject'] = kwargs['reject'] if 'reject' in kwargs else []
        self['stats'] = kwargs['stats'] if 'stats' in kwargs else []
        self['specdata'] = kwargs['specdata'] if 'specdata' in kwargs else []
        self['specicaact'] = kwargs['specicaact'] if 'specicaact' in kwargs else []
        self['splinefile'] = kwargs['splinefile'] if 'splinefile' in kwargs else []
        self['icasplinefile'] = kwargs['icasplinefile'] if 'icasplinefile' in kwargs else []
        self['dipfit'] = kwargs['dipfit'] if 'dipfit' in kwargs else []
        self['history'] = kwargs['history'] if 'history' in kwargs else []
        self['saved'] = kwargs['saved'] if 'saved' in kwargs else []
        self['etc'] = kwargs['etc'] if 'etc' in kwargs else []
        self['datfile'] = kwargs['datfile'] if 'datfile' in kwargs else []
        self['run'] = kwargs['run'] if 'run' in kwargs else []
    

if __name__=='__main__':
    validate_eeglab('/home/takemoto/EEG/eeglab2019_1')
