# LIA RISTIANA 2018
# This file calculates the t_score to compare the significance of difference
# between the actual and predicted values of reviews in each loop.

import pandas as pd
import numpy as np

import scipy.stats as st
from scipy.stats import ttest_ind_from_stats

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

def calculate_rating(df):
    i_length = len(df['iteration'].value_counts())

    for i in range(0, i_length):
        print("ITERATION-%d"%i)

        current_matrix = df[df['iteration'] == i]
        current_matrix = current_matrix.drop('iteration', 1)
        current_matrix.index = np.arange(1, len(current_matrix) + 1)
        print(current_matrix)

        # GET ACTUAL SCORE BY LOOPING THROUGH THE ROWS
        actual_list = []
        for ind, row in current_matrix.iterrows():
            row_sum = row.sum()
            row_value = row_sum * ind
            # print("%d x %d = %d"%(row_sum, ind, row_value))

            actual_list.append(row_value)

        sum_actual = sum(actual_list)

        # GET PREDICTION SCORE BY LOOPING THROUGH THE columns
        prediction_list = []
        for colname, col in current_matrix.iteritems():
            column_sum = col.sum()
            column_value = int(colname) * column_sum
            # print("%d x %d = %d"%(column_sum, int(colname), column_value))

            prediction_list.append(column_value)

        sum_prediction = sum(prediction_list)

        t_score, prob = calculate_t_score(prediction_list, actual_list)

# SUM ALL MATRICES
def sum_all_matrices(matrix_df):
    # print(matrix_list[matrix_list['iteration']==i] for i in matrix_list[['iteration']].nunique())
    matrices = matrix_df[['1', '2', '3', '4', '5', 'iteration']]

    d = matrices.set_index('iteration')
    sum_matrix = np.sum(d.loc[i].values for i in d.index.drop_duplicates().values)
    
    print(sum_matrix)
    avg_accu = sum(np.diag(sum_matrix))/sum_matrix.sum()
    print(avg_accu)

    matrix_df = pd.DataFrame(sum_matrix, columns=[1,2,3,4,5])
    # matrix_df.index = np.arange(1, len(matrix_df) + 1)
    # matrix_df['iteration'] = 100
    return matrix_df

def main():
    input_dir = '/home/lia/Documents/the_project/output/2018-05-20/matrices/'
    list_files = os.listdir(input_dir)

    # FOR EVERY FILE IN ABOVE DIRECTORY, READ THE CONTENT AND DO CALCULATION
    for i in range(0, len(list_files)):
        input_file = input_dir+list_files[i]
        print(list_files[i])
        matrices_df = pd.read_csv(input_file)
        print(matrices_df)

        sum_all_matrices(matrices_df)



    # calculate_rating(nb_df)

if __name__ == "__main__":
    main()
