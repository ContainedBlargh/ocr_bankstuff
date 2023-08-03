# %% Imports
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# %% Setup tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# %% encode training targets
def read_file(file):
    with open(file) as fp:
        return fp.read()

texts = None
with Pool(16) as pool:
    files = glob("fake_data/output_*.gt.txt")
    texts = list(pool.map(read_file, files))

encoded = tokenizer(texts, padding=True, truncation=True, max_length=1928, return_tensors="pt")
print(encoded['input_ids'])
