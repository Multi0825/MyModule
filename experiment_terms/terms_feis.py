# FEIS実験条件
n_sub = 22 # 被験者数
subs = ['01', '02', '03', '04', '05', '06', '07', '08',
        '09', '10', '11', '12', '13', '14', '15', '16',
        '17', '18', '19', '20', '21'] # 被験者(ファイル名)
labels = ['m', 'n', 'ng', 'f', 's', 'sh', 'v', 'z', 
          'zh', 'p', 't', 'k', 'fleece', 'goose', 'trap', 'thought'] # ラベル
tasks = ['vowel', 'voice', 'u'] # 論文中言及された2分類タスク(backもあるが結果が載ってない)
# 各ラベルとタスクの対応
labels4tasks = {'m':{'vowel':0,'voice':1,'u':0}, 'n':{'vowel':0,'voice':1,'u':0}, 
                'ng':{'vowel':0,'voice':1,'u':0}, 'f':{'vowel':0,'voice':0,'u':0}, 
                's':{'vowel':0,'voice':0,'u':0}, 'sh':{'vowel':0,'voice':0,'u':0}, 
                'v':{'vowel':0,'voice':1,'u':0}, 'z':{'vowel':0,'voice':1,'u':0},
                'zh':{'vowel':0,'voice':1,'u':0}, 'p':{'vowel':0,'voice':0,'u':0}, 
                't':{'vowel':0,'voice':0,'u':0}, 'k':{'vowel':0,'voice':0,'u':0},
                'fleece':{'vowel':1,'voice':1,'u':0}, 'goose':{'vowel':1,'voice':1,'u':1}, 
                'trap':{'vowel':1,'voice':1,'u':0}, 'thought':{'vowel':1,'voice':1,'u':0}} 
stages = ['stimuli', 'articulators', 'thinking', 'speaking', 'resting'] # ステージ
stage_ranges = {'stimuli':[0,5], 'articulators':[5,6], 'thinking':[6,11], 
               'speaking':[11,16],  'resting':[16, 21]} # ステージ範囲



