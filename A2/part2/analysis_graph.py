import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("clustering_values.csv")

df4 = df.groupby("dimentions").get_group(4)
df5 = df.groupby("dimentions").get_group(5)
df6 = df.groupby("dimentions").get_group(6)
df7 = df.groupby("dimentions").get_group(7)

dfz = df7.groupby("k").mean()
print(dfz)

plt.plot(dfz["distance"])
plt.show()