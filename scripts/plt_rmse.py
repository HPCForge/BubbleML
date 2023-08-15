import matplotlib.pyplot as plt
import numpy as np

def read_rmses(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        return [float(line) for line in lines]

saturated_rmses = read_rmses('scripts/maes')
#subcooled_rmses = read_rmses('scripts/subcooled_rmses')

plt.rc("font", family="serif", size=18, weight="bold")
plt.rc("axes", labelweight="bold")

plt.plot(range(len(saturated_rmses)), saturated_rmses, label='PB Saturated', linewidth=3)
plt.plot(range(len(saturated_rmses)), saturated_rmses, label='PB Subcooled', linewidth=3)
plt.xlabel('Iteration')
plt.ylabel('RMSE', labelpad=-80)
plt.legend(fontsize='16')
plt.savefig('iter_rmse.png', bbox_inches='tight', dpi=500)
