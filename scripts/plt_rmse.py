import matplotlib.pyplot as plt
import numpy as np
import torch

def read_tensors(path):
    temp_pred = torch.load(path + 'model_ouput.pt')
    temp_label = torch.load(path + 'sim_ouput.pt')

    velx_pred = torch.load(path + 'velx_output.pt')
    velx_label = torch.load(path + 'velx_label.pt')

    vely_pred = torch.load(path + 'vely_output.pt')
    vely_label = torch.load(path + 'vely_label.pt')

    temp_rmses = []
    vel_rmses = []
    velx_rmses = []
    vely_rmses = []

    for i in range(temp_pred.shape[0] // 2):
        temp_rmse = torch.sqrt(torch.mean((temp_pred[i] - temp_label[i]) ** 2))

        velx_rmse = torch.sqrt(torch.mean((velx_pred[i] - velx_label[i]) ** 2))
        vely_rmse = torch.sqrt(torch.mean((vely_pred[i] - vely_label[i]) ** 2))

        vel_pred = torch.stack((velx_pred[i], vely_pred[i]), dim=0)
        vel_label = torch.stack((velx_label[i], vely_label[i]), dim=0)
        vel_rmse = torch.sqrt(torch.mean((vel_pred - vel_label) ** 2).sum(dim=0))

        temp_rmses.append(temp_rmse)
        velx_rmses.append(velx_rmse)
        vely_rmses.append(vely_rmse)
        vel_rmses.append(vel_rmse)

    return temp_rmses, velx_rmses, vely_rmses, vel_rmses

t1, vx1, vy1, v1 = read_tensors('scripts/data/long_rollout/')
t2, vx2, vy2, v2 = read_tensors('scripts/data/long_rollout_push/')

plt.rc("font", family="serif", size=18, weight="bold")
plt.rc("axes", labelweight="bold")

plt.plot(range(len(t1)), t1, label='UNet$_{mod}$', linewidth=3, color='g')
plt.plot(range(len(t2)), t2, label='P-UNet$_{mod}$', linewidth=3, color='r')
plt.xlabel('Iteration')
plt.ylabel('RMSE', labelpad=-75)
plt.legend(fontsize='16')
plt.savefig('temp_iter_rmse.png', bbox_inches='tight', dpi=500)
plt.close()

plt.plot(range(len(t1)), vx1, label='UNet$_{mod}$', linewidth=3)
plt.plot(range(len(t2)), vx2, label='P-UNet$_{mod}$', linewidth=3)
plt.xlabel('Iteration')
plt.ylabel('RMSE', labelpad=-80)
plt.legend(fontsize='16')
plt.savefig('velx_iter_rmse.png', bbox_inches='tight', dpi=500)
plt.close()

plt.plot(range(len(t1)), vy1, label='UNet$_{mod}$', linewidth=3)
plt.plot(range(len(t2)), vy2, label='P-UNet$_{mod}$', linewidth=3)
plt.xlabel('Iteration')
plt.ylabel('RMSE', labelpad=-80)
plt.legend(fontsize='16')
plt.savefig('vely_iter_rmse.png', bbox_inches='tight', dpi=500)
plt.close()

plt.plot(range(len(t1)), v1, label='UNet$_{mod}$', linewidth=3, color='g')
plt.plot(range(len(t2)), v2, label='P-UNet$_{mod}$', linewidth=3, color='r')
plt.xlabel('Iteration')
plt.ylabel('RMSE', labelpad=-75)
plt.legend(fontsize='16')
plt.savefig('vel_iter_rmse.png', bbox_inches='tight', dpi=500)
plt.close()
