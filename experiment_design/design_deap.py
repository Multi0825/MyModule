n_sub = 32
subs = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08',
        's09', 's10', 's11', 's12', 's13', 's14', 's15', 's16',
        's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 
        's25', 's26', 's27', 's28', 's29', 's30', 's31', 's32']
labels = ['Valence', 'Arousal', 'Dominance', 'Liking']
score_range=[1,9]
states = ['LL', 'LH', 'HL', 'HH', 'All']
va_range = {'LL':((None,5),(None,5)), 
            'LH':((None,5),(4.999,None)),
            'HL':((4.999,None),(None,5)),
            'HH':((4.999,None),(4.999,None)),
            'All':((None, None),(None,None))}