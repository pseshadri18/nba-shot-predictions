import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(filename):
    return pd.read_csv(filename, sep=',')

def main():
    df = load_data("shotlogs.csv")
    counts = df['player_id'].value_counts()
    print counts.mean()
    num = 0
    ids = []
    for key, val in counts.iteritems():
        if val < 150:
            ids.append(key)
    print ids

    df = df[~df.player_id.isin(ids)]
    df = df[df.SHOT_DIST <= 45]
    df = df[(df.TOUCH_TIME > 0) & (df.TOUCH_TIME < 24)]
    df = df[df.PERIOD <= 4]
    df = df[np.isfinite(df['SHOT_CLOCK'])]
    print 'the length of the df is', len(df)

    seconds = lambda x : int(x.split(':')[0])*60 + int(x.split(':')[1])
    df['GAME_CLOCK'] = df['GAME_CLOCK'].apply(seconds) 
    df_subset = df[['SHOT_NUMBER', 'PERIOD', 'GAME_CLOCK', 'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'CLOSE_DEF_DIST', 'FGM']]

    # PLOTTING DISTANCE
    bins = [-0.1, 9, 18, 27, 36, 45]
    df_subset['categories'] = pd.cut(df_subset['SHOT_DIST'], bins)
    counts = df_subset.groupby('categories', as_index=False)['FGM'].mean()
    y_pos = np.arange(5)
    plt.bar(y_pos, counts['FGM'], align='center', alpha=0.5)
    plt.xticks(y_pos, counts['categories'])
    plt.ylabel('Shot Success')
    plt.title('Shot Success vs. Distance')
    plt.show()

    # PLOTTING PTS TYPE
    counts = df_subset.groupby('PTS_TYPE', as_index=False)['FGM'].mean()
    y_pos = np.arange(2)
    plt.bar(y_pos, counts['FGM'], align='center', alpha=0.5)
    plt.xticks(y_pos, counts['PTS_TYPE'])
    plt.ylabel('Shot Success')
    plt.title('Shot Success for 2 vs. 3 pointers')
    plt.show()

    # PLOTTING CLOSEST DEFENDER
    bins = [-0.1, 1, 2, 3, 4, 5, 6, 7, 53.2]
    df_subset['categories'] = pd.cut(df_subset['CLOSE_DEF_DIST'], bins)
    counts = df_subset.groupby('categories', as_index=False)['FGM'].mean()
    y_pos = np.arange(8)
    plt.bar(y_pos, counts['FGM'], align='center', alpha=0.5)
    plt.xticks(y_pos, counts['categories'])
    plt.ylabel('Shot Success')
    plt.title('Shot Success vs. Closest Defender')
    plt.show()

    # PLOTTING TOUCH TIME
    bins = [-0.1, 1, 2, 4, 6, 8, 16, 20, 25]
    df_subset['categories'] = pd.cut(df_subset['TOUCH_TIME'], bins)
    counts = df_subset.groupby('categories', as_index=False)['FGM'].mean()
    print counts
    y_pos = np.arange(8)
    print y_pos
    plt.bar(y_pos, counts['FGM'], align='center', alpha=0.5)
    plt.xticks(y_pos, counts['categories'])
    plt.ylabel('Shot Success')
    plt.title('Shot Success vs. Touch Time')
    plt.show()

    # PLOTTING DRIBBLES
    bins = [-0.1, 1, 2, 3, 4, 5, 32]
    df_subset['categories'] = pd.cut(df_subset['DRIBBLES'], bins)
    counts = df_subset.groupby('categories', as_index=False)['FGM'].mean()
    print counts
    y_pos = np.arange(6)
    print y_pos
    plt.bar(y_pos, counts['FGM'], align='center', alpha=0.5)
    plt.xticks(y_pos, counts['categories'])
    plt.ylabel('Shot Success')
    plt.title('Shot Success vs. No. of Dribbles')
    plt.show()

    # PLOTTING GAME CLOCK
    bins = [-0.1, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720]
    df_subset['categories'] = pd.cut(df_subset['GAME_CLOCK'], bins)
    counts = df_subset.groupby('categories', as_index=False)['FGM'].mean()
    print counts
    y_pos = np.arange(12)
    print y_pos
    plt.bar(y_pos, counts['FGM'], align='center', alpha=0.5)
    plt.xticks(y_pos, counts['categories'],fontsize=6)
    plt.ylabel('Shot Success')
    plt.title('Shot Success vs. Time on Game Clock')
    plt.show()

    # PLOTTING SHOT CLOCK
    bins = [-0.1, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24]
    df_subset['categories'] = pd.cut(df_subset['SHOT_CLOCK'], bins)
    counts = df_subset.groupby('categories', as_index=False)['FGM'].mean()
    print counts
    y_pos = np.arange(12)
    print y_pos
    plt.bar(y_pos, counts['FGM'], align='center', alpha=0.5)
    plt.xticks(y_pos, counts['categories'], fontsize=6)
    plt.ylabel('Shot Success')
    plt.title('Shot Success vs. Time on Shot Clock')
    plt.show()

    # PLOTTING PERIOD
    bins = [0, 1, 2, 3, 4]
    df_subset['categories'] = pd.cut(df_subset['PERIOD'], bins)
    counts = df_subset.groupby('categories', as_index=False)['FGM'].mean()
    print counts
    y_pos = np.arange(4)
    print y_pos
    plt.bar(y_pos, counts['FGM'], align='center', alpha=0.5)
    plt.xticks(y_pos, counts['categories'])
    plt.ylabel('Shot Success')
    plt.title('Shot Success vs. Period')
    plt.show()

    # PLOTTING SHOT NO.
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    df_subset['categories'] = pd.cut(df_subset['SHOT_NUMBER'], bins)
    counts = df_subset.groupby('categories', as_index=False)['FGM'].mean()
    print counts
    y_pos = np.arange(8)
    print y_pos
    plt.bar(y_pos, counts['FGM'], align='center', alpha=0.5)
    plt.xticks(y_pos, counts['categories'])
    plt.ylabel('Shot Success')
    plt.title('Shot Success vs. Shot No.')
    plt.show()
   

if __name__ == "__main__":
    main()