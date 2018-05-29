# import numpy
import pandas
# import matplotlib.pyplot as plt
import random

import time
start_time = time.time()

input_file='/home/lia/Documents/the_project/dataset/movies_tv_reviews_sm.csv'

df = pandas.read_csv(input_file)

pandas.set_option('float_format', '{:f}'.format)

print("DATA SHAPE")
print(df.shape)
#
# df['class'] = df['class'].astype('category')
print("COLUMN TYPES")
print(df.dtypes)
#


# DATA STATS
print("DATA STATS")
print(df["overall"].describe())

# asinCount = df['asin'].value_counts(sort=False)
# print("count: ")
# print(asinCount)


# dataStats = df.describe()
# print(dataStats)

# df.describe().to_csv("results/statistic.csv", encoding="utf-8", sep=",")
#
# classCount = df['class'].value_counts(sort=False)
# print(classCount)
#

#
# # # accessing series
# # print(classCount.values)
# # print(classCount.index.tolist())
#
# plt.bar(classCount.index.tolist(),classCount.values)
# plt.xlabel('class')
# plt.ylabel('freq')
# plt.title('Class frequency')
# plt.savefig("results/class-frequency.png")
# # plt.show()


time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
