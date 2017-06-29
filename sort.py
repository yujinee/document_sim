import re
import yaml
from operator import itemgetter

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.strip().split() for line in f.read().splitlines()]
        data = [d[1:-1] for d in data]
    return data

def read_dictionary(filename):
    with open(filename, 'r') as f:
        data = f.readline()
        return data


d = yaml.load(read_dictionary('dictionary.dict'))

lst = [(k, v) for k, v in sorted(d.items(), key=itemgetter(1), reverse=True)]

print(lst)
