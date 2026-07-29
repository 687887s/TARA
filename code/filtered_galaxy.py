import pandas as pd
import glob
files = glob.glob('../data/catalog/ps1/redshift_STRM/*.csv')
print(len(files))
for i, file in enumerate(files):
    df = pd.read_csv(file)
    print(len(df), 'origin')
    try:
        df = df[df.z_phot >= 0]
        df = df[df.prob_Galaxy > df.prob_Star]
        df = df[df.prob_Galaxy > df.prob_QSO]
        df = df[df.z_phot != -999.0]
    except:
        print('', end = '')
    try:
        df = df.drop(columns = ['prob_Galaxy', 'prob_Star', 'prob_QSO'])
    except:
        print('', end= '')
    try:
        for idx, col in enumerate(df.columns):
            if col == 'objID':
                break
        df = df.drop(columns = df.columns[:idx])
        df = df.reset_index(drop=True)
        df.to_csv(file)
    except Exception as e:
        print(e)
    print(len(df), 'after')
