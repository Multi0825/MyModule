# 実験条件
subs = ['Sub1','Sub2','Sub3'] # 被験者
tasks = ['J1', 'J3', 'J5','E'] # タスク
n_session = 4 # セッション数
j_labels = ['a', 'i', 'u', 'e', 'o'] # 日音素ラベル
e_labels = ['k', 's', 't', 'n'] # 英音素ラベル
j_stages = ['resting', 'stimuli','preparing1','thinking'] # 日音素ステージ
e_stages = ['resting', 'stimuli','preparing1','thinking', 'preparing2', 'speaking'] # 英音素ステージ
j_trial_length = 16 # 日音素合計時間
e_trial_length = 22 # 英音素合計時間
stage_ranges = {'resting':[0,5], 'stimuli':[5,10], 'preparing1':[10,11], \
               'thinking':[11,16], 'preparing2':[16, 17], 'speaking':[17, 22]} # ステージ音素時間範囲
tasks_labels = {'J1':j_labels, 'J3':j_labels, 'J5':j_labels,'E':e_labels} # タスクラベル変換
tasks_stages = {'J1':j_stages, 'J3':j_stages, 'J5':j_stages,'E':e_stages} # タスクステージ変換
task_length = {'J1':j_trial_length,'J3':j_trial_length,'J5':j_trial_length,'E':e_trial_length}

# 追加(FEIS, KARAONEと比較、連携用)
# FEISとKARAONEと共有できるタスク(このタスクは分類タスク)
common_tasks = ['voice', 'nasal'] # voice(FEIS):有声、nasal(KARAONE):鼻音
# ※一応日本語でFEISの±uもあるが、FEISは英語だから一旦なし

# 各ラベルとタスクの対応
labels4tasks = {'k':{'voice':0,'nasal':0}, 's':{'voice':0,'nasal':0}, 
                't':{'voice':0,'nasal':0}, 'n':{'voice':1,'nasal':1}} # 英音素に限定 


