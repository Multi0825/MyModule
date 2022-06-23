n_sub = 32
subs = ['01', '02', '03', '04', '05', '06', '07', '08',
        '09', '10', '11', '12', '13', '14', '15', '16',
        '17', '18', '19', '20', '21','22', '23', '24', 
        '25', '26','27', '28', '29', '30', '31', '32']
labels = ['Valence', 'Arousal', 'Dominance', 'Liking']
score_range=[1,9]
states = ['LL', 'LH', 'HL', 'HH', 'All']
va_range = {'LL':((None,5),(None,5)), 
            'LH':((None,5),(4.999,None)),
            'HL':((4.999,None),(None,5)),
            'HH':((4.999,None),(4.999,None)),
            'All':((None, None),(None,None))}