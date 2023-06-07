import matplotlib.pyplot as plt
import numpy as np

plt.rc("font", family="serif", size=18, weight="bold")
plt.rc("axes", labelweight="bold")
#plt.rc("text", usetex=True)
#plt.figure(figsize=(5, 4), dpi=400)

t_wall_subcooled = [79, 81, 85, 90, 95, 98, 100, 103, 106, 110]
y1 = [11823.9753, 17338.4293, 17819.4369, 23312.0327, 26425.6244, 28200.2418, 30611.4471, 34459.4348, 36112.5162, 38661.8841]

t_wall_saturated = [65, 70, 75, 80, 85, 90, 95, 100]#, 150]
y2 = [3384.0490, 7500.3820, 9104.2976, 13099.5624, 15099.1912, 17036.5389, 18199.7169, 20111.4978]#, 24238.6561]

x1 = [(x-58) for x in list(sorted(t_wall_subcooled))]
x2 = [(x-58) for x in list(sorted(t_wall_saturated))]
y1 = np.array(y1)/np.max(y1)

# hard-code a max HF value, since we don't actually cover
# the full range in the cross validation.
# This max is approximately the expected heatflux and is used
# for normalization only
y2 = np.array(y2)/23000 

plt.scatter(x1, y1, color='red', label='Subcooled Prediction')
plt.scatter(x2, y2, color='blue', marker='^', label='Saturated Prediction')
expected_x1 = [20, 25, 30, 35, 40, 45, 50]
expected_y1 = [0.3, 0.5, 0.6, 0.72, 0.8, 0.9, 1.0]
expected_x2 = [7, 12, 22, 27, 37, 52]
expected_y2 = [0.2, 0.4, 0.6, 0.7, 0.8, 1.0]
plt.plot(expected_x1, expected_y1, '--', color='red', label='Subcooled Expected')
plt.plot(expected_x2, expected_y2, '--', color='blue', label='Saturated Expected')
plt.xticks(np.arange(0, 65, 5))
plt.yticks(np.arange(0, 1.1, 0.2))
plt.xlabel(r'$T_{wall} - T_{sat}(^\circ C)$')
plt.ylabel(r'${q}/{q_{max}}$', rotation='horizontal', labelpad=-80)
plt.legend(loc='lower right', fontsize="16")
plt.savefig('boiling_curve.png', bbox_inches="tight", dpi=500)
plt.show()
