import csv, json, sys

input = open('output.strict')
data = json.load(input)
input.close()

output = csv.writer(sys.stdout)

output.writerow(data[0].keys())  # header row

for row in data:
    output.writerow(row.values())
