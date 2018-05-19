# LIA RISTIANA 2018
# This file calculates the t_score

import os

def calculate_t_score():
    return

def main():
    # input_file = '/home/lia/Documents/the_project/dataset/to_use/current/top_30_clean.csv'
    input_dir = '/home/lia/Documents/the_project/output/2018-05-19/data/'
    list_files = os.listdir(input_dir)

    for i in range(0, len(list_files)):
        input_file = input_dir+list_files[i]
        

    # df_prediction = pd.read_csv(input_file)
    # calculate_t_score()

    # SET ORDER listdir BY DATE PLEASE
if __name__ == "__main__":
    main()
