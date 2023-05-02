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


# companies = {}
# min_date = datetime.datetime.max
# max_date = datetime.datetime.min
# for file in os.listdir("data"):
#     comp = "data/" + file
#     if os.stat(comp).st_size == 0: continue
#     df = pd.read_csv(comp)
#     df = df.drop(columns=["Open", "Close", "Volume", "OpenInt"])
#     df["Price"] = (df["High"] + df["Low"]) / 2
#     df = df.drop(columns=["High", "Low"])
#     start_date = pd.to_datetime(df.loc[0,"Date"])
#     end_date = pd.to_datetime(df.iloc[-1,df.columns.get_loc("Date")])
#     if start_date < min_date: min_date = start_date
#     if end_date > max_date: max_date = end_date
#     companies[file[:-7]] = df


# dates = defaultdict(set)
# for company, df in companies.items():
#     for date in df["Date"]:
#         dates[date].add(company)

# jsondates = jsonpickle.encode(dates)
# with open("dates.txt","w") as f:
#     f.write(json.dumps(jsondates))

with open("dates.txt","r") as f:
    jsondata = f.read()
dates = jsonpickle.decode(json.loads(jsondata))

min_date = datetime.datetime(1962,1,2)
max_date = datetime.datetime(2017,1,2)
date_range = pd.date_range(start=min_date, end=max_date, freq='D')

n = []
t = []
delta_t = []
delta = (max_date-min_date).days

npoints = 0
while npoints < 1000:
    print(npoints)
    random_days_1 = random.randint(0, delta)
    random_date_1 = min_date + timedelta(days=random_days_1)
    random_days_2 = random.randint(0, delta-random_days_1)
    random_date_2 = random_date_1 + timedelta(days=random_days_2)
    
    point = len(dates[random_date_1].intersection(dates[random_date_2]))
    if point != 0:
        t.append(random_date_1)
        delta_t.append(random_days_2)
        n.append(point)
        npoints += 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = [datetime.datetime.timestamp(date) for date in t]
y = delta_t
z = n

ax.scatter(x, y, z)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda timestamp, _: datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')))
plt.show()
