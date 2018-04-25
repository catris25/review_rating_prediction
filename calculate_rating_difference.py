import pandas as pd
import numpy as np

def calculate_rating(df):
    i_length = len(df['iteration'].value_counts())

    for i in range(0, i_length):
        current_matrix = df[df['iteration'] == i]
        current_matrix = current_matrix.drop('iteration', 1)
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

        # print("ITERATION-%d"%i)

        # CALCULATE THE DIFFERENCE
        # print("actual score:",sum_actual)
        # print("prediction score:",sum_prediction)
        # diff = abs(sum_actual - sum_prediction)
        # print(diff)

        if sum_actual > sum_prediction:
            print("ACTUAL SCORE")
        elif sum_prediction > sum_actual:
            print("PREDICTION SCORE")
        else:
            print("EQUAL")


def main():
    file_folder = '/home/lia/Documents/the_project/output/main_result/'
    input_file = '/home/lia/Documents/the_project/output/main_result/p_smote_logreg.csv'
    nb_df = pd.read_csv(input_file, index_col=0)

    calculate_rating(nb_df)

    # NOTE TO SELF
    # CALCULATE THE Z TEST
    # THERE IS SOMETHING WRONG WITH SMOTE LOGREG PLS CHECK

if __name__ == "__main__":
    main()
