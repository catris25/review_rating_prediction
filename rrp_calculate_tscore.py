# LIA RISTIANA 2018
# This file calculates the t_score to compare the significance of difference
# between the actual and predicted values of a film.

import pandas as pd
import numpy as np
import scipy.stats as st

import os

# CALCULATE THE T-SCORE AND PROBABILITY VALUE
def calculate_t_score(pop_list, samp_list):
    n1 = len(samp_list)
    n2 = len(pop_list)

    mean1 = np.mean(samp_list)
    mean2 = np.mean(pop_list)

    std1 = np.std(samp_list, ddof=1)
    std2 = np.std(pop_list, ddof=1)
    var1 = std1**2
    var2 = std2**2

    standard_error = np.sqrt(((n1-1)*var1+(n2-1)*var2)/(n1+n2-2))
    t_score = (mean1 - mean2)/(standard_error*(np.sqrt((1/n1)+(1/n2))))
    prob = st.t.sf(np.abs(t_score), df=4)

    print("t :",t_score)
    print("p :",prob)

    return(t_score, prob)


# CALCULATE SCORE PER FILM
def compare_film_scores(review_scores_df, df):
    # MAP THE REVIEW SCORES TO ITS CORRESPONDING review_ids
    new_df = df[['review_id', 'asin', 'overall']].merge(review_scores_df, left_on='review_id', right_on='review_id')
    new_df = new_df.rename(columns = {'overall':'actual'})

    # CALCULATE THE AVERAGE RATING PER FILM BASED ON ITS PREDICTED SCORES
    avg_prediction = new_df['prediction'].groupby(new_df['asin']).mean().reset_index()
    df_avg_prediction = pd.DataFrame(avg_prediction)

    # CALCULATE THE AVERAGE RATING PER FILM BASED ON ITS ACTUAL RATINGS
    avg_actual = new_df['actual'].groupby(new_df['asin']).mean().reset_index()
    df_avg_actual = pd.DataFrame(avg_actual)

    # MERGE BOTH RATINGS FOR COMPARISON
    df_comparison = pd.merge(df_avg_prediction, df_avg_actual, on="asin")
    print(df_comparison)

    return df_comparison

def main():
    # input_file = '/home/lia/Documents/the_project/dataset/to_use/current/top_30_clean.csv'
    input_dir = '/home/lia/Documents/the_project/output/2018-05-20/data/'
    list_files = os.listdir(input_dir)

    input_file = '/home/lia/Documents/the_project/output/2018-05-20/df.csv'
    df = pd.read_csv(input_file)

    # FOR EVERY FILE IN ABOVE DIRECTORY, READ THE CONTENT AND DO CALCULATION
    for i in range(0, len(list_files)):
        input_file = input_dir+list_files[i]
        print(list_files[i])
        pred_df = pd.read_csv(input_file)

        # CALCULATE SCORE PER REVIEW ON AVERAGE
        review_scores = pred_df[['review_id', 'prediction']].groupby('review_id')['prediction'].mean().reset_index()
        df_comparison = compare_film_scores(review_scores, df)

        # CONVERT DF TO LISTS
        prediction_list = df_comparison['prediction'].tolist()
        actual_list = df_comparison['actual'].tolist()
        
        t_score, prob = calculate_t_score(prediction_list, actual_list)


    # SET ORDER listdir BY DATE PLEASE
if __name__ == "__main__":
    main()
