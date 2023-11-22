import sys
import numpy as np
from matplotlib import pyplot as plt

def simple_progress_bar(total_iterations, current_index):
    progress = (current_index / total_iterations) * 100
    sys.stdout.write(f"\r[{('=' * (int(progress) // 10)) + (' ' * (10 - (int(progress) // 10)))}] {int(progress)}%")
    sys.stdout.flush()
    
def mean(iter):
    return sum(iter)/len(iter)

DIMENSIONS = [1, 2, 4, 8, 16, 32, 64]
L1_RATIOS = list()
L2_RATIOS = list()
LINF_RATIOS = list()
NUM_POINTS = int(1e6)
QUERY_POINTS = 100

for DIMENSION in DIMENSIONS:
    print(f"Processing {DIMENSION}")
    
    dataset = np.random.uniform(low=0.0, high=1.0, size=(NUM_POINTS, DIMENSION))
    random_points = np.random.choice(NUM_POINTS, QUERY_POINTS, replace=False)

    L1_min = list()
    L1_max = list()
    L2_min = list()
    L2_max = list()
    Linf_min = list()
    Linf_max = list()
    
    for i, random_idx in enumerate(random_points):
        simple_progress_bar(QUERY_POINTS, i)
        
        random_datapoint = np.reshape(dataset[random_idx, :], (1,-1))
        other_points = np.concatenate((dataset[:random_idx, :], dataset[random_idx+1:, :]), axis=0)
        
        L1 = np.sum(np.abs(random_datapoint - other_points), axis=1)
        L2 = np.sum((random_datapoint - other_points)**2, axis=1)**0.5
        Linf = np.max(np.abs(random_datapoint - other_points), axis=1)
        

        L1_min.append(np.min(L1))
        L1_max.append(np.max(L1))
        
        L2_min.append(np.min(L2))
        L2_max.append(np.max(L2))
        
        Linf_min.append(np.min(Linf))
        Linf_max.append(np.max(Linf))
    
    L1_ratio =   [x/y for x, y in zip(L1_max, L1_min)]
    L2_ratio =   [x/y for x, y in zip(L2_max, L2_min)]
    Linf_ratio = [x/y for x, y in zip(Linf_max, Linf_min)]
    
    L1_RATIOS.append(mean(L1_ratio))
    L2_RATIOS.append(mean(L2_ratio))
    LINF_RATIOS.append(mean(Linf_ratio))

    print()
    print(f"Dim: {DIMENSION}, L1:   Min: {mean(L1_min):.5f},   L1: Max: {mean(L1_max):.5f}, L1 Max-Min Ratio: {mean(L1_ratio):.5f}")
    print(f"Dim: {DIMENSION}, L2:   Min: {mean(L2_min):.5f},   L2: Max: {mean(L2_max):.5f}, L2 Max-Min Ratio: {mean(L2_ratio):.5f}")
    print(f"Dim: {DIMENSION}, Linf: Min: {mean(Linf_min):.5f}, Linf: Max: {mean(Linf_max):.5f}, Linf Max-Min Ratio: {mean(Linf_ratio):.5f}")


plt.plot(DIMENSIONS, L1_RATIOS, marker='x', label="L1 Distance")
plt.plot(DIMENSIONS, L2_RATIOS, marker='^', label="L2 Distance")
plt.plot(DIMENSIONS, LINF_RATIOS, marker='o', label="Linf Distance")
plt.yscale('log')

plt.xlabel('Dimension')
plt.ylabel('Average of (Max/Min) Distance (Log-Scale)')
plt.title('Curse of Dimensionality')
plt.legend()
plt.grid()

plt.savefig('Q1.png')
