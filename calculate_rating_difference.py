import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.stats import ttest_ind_from_stats

def calculate_t_score(pop_list, samp_list):
    n1 = len(samp_list)
    n2 = len(pop_list)

    mean1 = np.mean(samp_list)
    mean2 = np.mean(pop_list)

    var1 = np.var(samp_list,ddof=1)
    # var1 = sum([(xi - mean1) ** 2 for xi in samp_list]) / (n1 - 1)
    var2 = np.var(pop_list, ddof=1)

    # print("VAR1",var1)
    # print("VAR2",var2)

    std1 = np.std(samp_list, ddof=1)
    std2 = np.std(pop_list, ddof=1)

    standard_error = np.sqrt(var1/n1 + var2/n2)
    # standard_error = std1/np.sqrt(n1) + std2/np.sqrt(n2)
    t_score = (mean1 - mean2)/standard_error
    t_score = t_score * 2

    prob = st.t.sf(np.abs(t_score), df=4)

    print("t :",t_score)
    print("p :",prob)

    # tstat, pvalue = ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2)
    # print(pvalue)

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


def main():
    input_file = '/home/lia/Documents/the_project/output/main_result/sum.csv'
    nb_df = pd.read_csv(input_file, index_col=0)

    calculate_rating(nb_df)

if __name__ == "__main__":
    main()
