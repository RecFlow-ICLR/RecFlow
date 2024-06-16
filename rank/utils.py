import pickle as pkl
from collections import defaultdict

def defaultdict_tuple():
  return defaultdict(tuple)

def defaultdict_str():
  return defaultdict(str)

def load_pkl(filename):
  with open(filename, 'rb') as f:
    return pkl.load(f)