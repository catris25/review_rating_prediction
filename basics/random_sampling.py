import pandas as pd

local_dir = '/home/lia/Documents/the_project/dataset/to_use/helpfulness/'

df = pd.read_csv(local_dir+'helpful_en.csv')

df.rename(columns={'Unnamed: 0':'review_id'}, inplace=True)

asin_list = df['asin'].value_counts().index.tolist()

for i in range(0,30):
    df_stratified = pd.DataFrame()

    fraction = 0.1
    for name in asin_list:
        temp = df.loc[df['asin']==name].sample(frac=fraction)
        df_stratified = pd.concat([df_stratified,temp])

    df_stratified.to_csv(('%ssamples/10percent/%s.csv'%(local_dir,i)), sep=",", encoding="utf-8", index=False)
    print("done",i)
