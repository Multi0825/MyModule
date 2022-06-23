# KARAONE実験条件
n_subs = 14 # 被験者数
subs = ['MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14', 
        'MM15', 'MM16', 'MM18', 'MM19', 'MM20', 'MM21', 'P02'] # 被験者(ファイル名)
labels = ['iy', 'uw', 'm', 'n', 'piy', 'tiy',
          'diy', 'gnaw', 'knew', 'pat', 'pot'] # ラベル
# 論文中で言及された2分類タスク(nasal:鼻音、bilabial:両唇音)
tasks = ['vowel', 'nasal', 'bilabial', 'iy', 'uw'] 
# ラベルをタスク用に変換
labels4tasks = {'iy':{'vowel':1,'nasal':0,'bilabial':0,'iy':1,'uw':0}, 
                'uw':{'vowel':1,'nasal':0,'bilabial':0,'iy':0,'uw':1}, 
                'm':{'vowel':0,'nasal':1,'bilabial':1,'iy':0,'uw':0}, 
                'n':{'vowel':0,'nasal':1,'bilabial':0,'iy':0,'uw':0}, 
                'piy':{'vowel':1,'nasal':0,'bilabial':1,'iy':1,'uw':0}, 
                'tiy':{'vowel':1,'nasal':0,'bilabial':0,'iy':1,'uw':0},
                'diy':{'vowel':1,'nasal':0,'bilabial':0,'iy':1,'uw':0}, 
                'gnaw':{'vowel':1,'nasal':1,'bilabial':0,'iy':0,'uw':0}, 
                'knew':{'vowel':1,'nasal':1,'bilabial':0,'iy':0,'uw':0}, 
                'pat':{'vowel':1,'nasal':0,'bilabial':1,'iy':0,'uw':0}, 
                'pot':{'vowel':1,'nasal':0,'bilabial':1,'iy':0,'uw':0}}
stages = ['resting', 'stimuli', 'thinking', 'speaking'] # ステージ
