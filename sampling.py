import random

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[0:]
    return data

orig_data = read_data('misspelled_170314.txt')
sample_data=[]
"""
for i in range(len(orig_data)):
    if random.randint(0,10) == 0:
        sample_data.append(orig_data[i])
"""

sample_data = orig_data[:100]

for i in sample_data:
    print(i[0])

