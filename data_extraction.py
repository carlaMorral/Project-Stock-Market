import os
import json
import jsonpickle
import random
import datetime
import numpy as np
import pandas as pd
from datetime import timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


with open("out.txt", "r") as f:
    stocks = f.readlines()

dfs = []
min_date = "2005-03-01"
max_date = "2016-12-30"
for file in os.listdir("data"):
    if file[:-7] in stocks:
        comp = "data/" + file
        if os.stat(comp).st_size == 0: continue
        df = pd.read_csv(comp)
        df = df.drop(columns=["Open", "Close", "Volume", "OpenInt"])
        df["Price"] = (df["High"] + df["Low"]) / 2
        df = df.drop(columns=["High", "Low"])
        df = df[(df["Date"] >= min_date) & (df["Date"] <= max_date)]
        df["Stock"] = [file[:-7]]*len(df)
        dfs.append(df)

print(dfs)
pd.concat(dfs).to_csv("data.csv")
