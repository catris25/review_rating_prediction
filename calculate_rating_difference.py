import pandas as pd
import numpy as np
import scipy.stats as st

def calculate_t_score(pop_list, samp_list):
    mean1 = np.mean(samp_list)
    mean2 = np.mean(pop_list)

    var1 = np.var(samp_list)
    var2 = np.var(pop_list)

    standard_error = np.sqrt(var1/len(samp_list)+var2/len(pop_list))
    t_score = (mean1-mean2)/standard_error
    prob = st.t.sf(np.abs(t_score),4)

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
        # print(current_matrix)

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

    # NOTE TO SELF
    # CALCULATE THE Z TEST
    # THERE IS SOMETHING WRONG WITH SMOTE LOGREG PLS CHECK

if __name__ == "__main__":
    main()
