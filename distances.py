import os
import pandas as pd


with open("data.csv", "r") as f:
    stocks = f.readlines()
stocks = set(map(lambda comp: comp[:-1], stocks))

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
