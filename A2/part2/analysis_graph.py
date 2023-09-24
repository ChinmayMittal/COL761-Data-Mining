import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--output_img', type=str, default='output.png', help='Name of the output image file')
parser.add_argument('--dim', '-d', type=int, help='dimensions')

args = parser.parse_args()

df = pd.read_csv("clustering_values.csv")


df = df.groupby("dimensions").get_group(args.dim)


dfz = df.groupby("k").mean()
print(dfz)

plt.plot(dfz["distance"], marker = 'o', color = 'r')
plt.title("Elbow Plot(dimension = {})".format(args.dim))
plt.xlabel("Number of clusters(k)")
plt.ylabel("Average distance from converged centers")
plt.grid(True)
plt.savefig(args.output_img)