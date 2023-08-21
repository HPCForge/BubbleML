
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import numpy as np

velx_pred = torch.load('scripts/data/vel_unet_mod_push/velx_output.pt').numpy()
vely_pred = torch.load('scripts/data/vel_unet_mod_push/vely_output.pt').numpy()

velx_label = torch.load('scripts/data/vel_unet_mod_push/velx_label.pt').numpy()
vely_label = torch.load('scripts/data/vel_unet_mod_push/vely_label.pt').numpy()

w = velx_pred.shape[1]
d = 6
y, x = np.mgrid[d:w-d,d:w-d]
print(x, y)

temp_ranges = [0.0, 0.1, 0.4, 0.75, 1.0]
color_codes = ['black', 'purple', 'orange', 'yellow', 'white']
colors = list(zip(temp_ranges, color_codes))
cmap = LinearSegmentedColormap.from_list('vel_colormap', colors)

steps = list(range(1, velx_label.shape[0] // 2 + 1, 20))

plt.rc("font", family="serif", size=18, weight="bold")
plt.rc("axes", labelweight="bold")
fig, ax = plt.subplots(3, len(steps), figsize=(14.5, 7))

mag_label = np.sqrt(velx_label**2 + vely_label**2)
mag_pred = np.sqrt(velx_pred**2 + vely_pred**2)
mag_error = np.abs(mag_label - mag_pred)

vmin, vmax = 0, max(mag_label[steps].max(), mag_pred[steps].max())

for idx, t in enumerate(steps):
    label_im = ax[0][idx].imshow(np.flipud(mag_label[t]), cmap=cmap, vmin=vmin, vmax=vmax)
    ax[0][idx].streamplot(x, y, velx_label[t,d:-d,d:-d], vely_label[t,d:-d,d:-d], linewidth=0.5, density=0.75, color='w', arrowstyle='fancy')
    ax[1][idx].imshow(np.flipud(mag_pred[t]), cmap=cmap, vmin=vmin, vmax=vmax)
    ax[1][idx].streamplot(x, y, velx_pred[t,d:-d,d:-d], vely_pred[t,d:-d,d:-d], linewidth=0.5, density=0.75, color='w', arrowstyle='fancy')
    error_im = ax[2][idx].imshow(np.flipud(mag_error[t]), cmap=cmap, vmin=vmin, vmax=mag_error[steps].max())
    for i in range(3):
        ax[i][idx].set_xticks([])
        ax[i][idx].set_yticks([])
ax[0][0].set_ylabel('Ground Truth')
ax[1][0].set_ylabel('Prediction')
ax[2][0].set_ylabel('Abs. Error')

for i in range(len(steps)):
    ax[0,i].set_title(f'Step {i*20}')

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
fig.colorbar(label_im, ax=ax[:2].ravel().tolist(), pad=0.005, shrink=0.5)
fig.colorbar(error_im, ax=ax[2].ravel().tolist(), pad=0.005)
plt.savefig(f'vel.png', dpi=500)
plt.close()
