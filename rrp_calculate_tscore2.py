# LIA RISTIANA 2018
# This file calculates the t_score to compare the significance of difference
# between the actual and predicted values of reviews in each loop.

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

    # print("t :",t_score)
    # print("p :",prob)

    return(t_score, prob)

def main():
    # input_file = '/home/lia/Documents/the_project/dataset/to_use/current/top_30_clean.csv'
    input_dir = '/home/lia/Dropbox/output/2018-07-02/data/'
    list_files = os.listdir(input_dir)

    input_file = '/home/lia/Dropbox/output/2018-07-02/df.csv'
    df = pd.read_csv(input_file)

    # FOR EVERY FILE IN ABOVE DIRECTORY, READ THE CONTENT AND DO CALCULATION
    for f in list_files:
        input_file = input_dir+f
        print("\n",f)
        pred_df = pd.read_csv(input_file)

        pred_df = df[['review_id', 'asin', 'overall']].merge(pred_df, left_on='review_id', right_on='review_id')
        pred_df = pred_df.rename(columns = {'overall':'actual'})

        significance_diff = []
        alpha = 0.05

        # FOR EVERY ITERATION, CALCULATE THE T-SCORE AND PROBABILITY
        for j in range(0, pred_df['iteration'].nunique()):
            current_df = pred_df[pred_df['iteration']==j]
            prediction_list = current_df['prediction'].tolist()
            actual_list = current_df['actual'].tolist()

            t_score, prob = calculate_t_score(prediction_list, actual_list)

            # CHECK PROB RELATIVE TO ALPHA
            if prob < alpha:
                significance_diff.append(True)
            else:
                significance_diff.append(False)

            # IF P IS VERY LOW (< alpha), REJECT THE NULL HYPOTHESIS
            # CONCLUDE THAT THERE IS A STATISTICALLY SIGNINIFICANT DIFFERENCE BETWEEN THE TWO DATA
            # IF P IS HIGH (>= alpha)
            # CONCLUDE THAT THERE IS NO STATISTICALLY SIGNINIFICANT DIFFERENCE

            # END OF LOOP

        print("is there any statistically significant difference?")
        print(significance_diff)

if __name__ == "__main__":
    main()
