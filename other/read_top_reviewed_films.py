import pandas as pd

# GET LIST OF FILMS
df_train = pd.read_csv('/home/lia/Dropbox/output/preprocessed.csv')
df_train = df_train.drop(['review_id', 'word'], axis=1)
list_of_films = df_train['asin'].unique()

input_file = '/home/lia/Documents/the_project/dataset/helpfulness/zero.csv'
df_zero = pd.read_csv(input_file)
df_zero = df_zero[df_zero['asin'].isin(list_of_films)]

input_file = '/home/lia/Documents/the_project/dataset/helpfulness/unhelpful.csv'
df_unhelpful = pd.read_csv(input_file)
df_unhelpful = df_unhelpful[df_unhelpful['asin'].isin(list_of_films)]

input_file = '/home/lia/Documents/the_project/dataset/helpfulness/helpful.csv'
df_helpful = pd.read_csv(input_file)
df_helpful = df_helpful[df_helpful['asin'].isin(list_of_films)]
df_helpful = df_helpful[~df_helpful.isin(df_train)].dropna()

print(len(df_zero))
print(len(df_unhelpful))
print(len(df_helpful))

print(df_zero['asin'].unique())
print(df_unhelpful['asin'].unique())
print(df_helpful['asin'].unique())
