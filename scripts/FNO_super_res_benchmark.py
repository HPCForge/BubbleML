import sys; sys.path.insert(0, '..')
from src.op_lib.metrics import interface_rmse
from src.op_lib.losses import LpLoss
import h5py
import numpy as np
import matplotlib.pyplot as plt
#import numba as nb
import torch
#from torch.nn.functional import interpolate
from neuralop.models import FNO
import torch.nn.functional as F
import re
#from neuralop.layers.resample import resample


#print("IMPORTS DONE")

future_window = 5
time_window = 5
n_layers = 6

checkpoint_path = "/dfs6/pub/nsankar1/res_exp/BubbleML/logs/24466485/wall_super_heat/FNO_temp_input_dataset_250_1696225650.pt"

twall_110_ds = '/share/crsp/lab/amowli/share/BubbleML2/PoolBoiling-WallSuperheat-FC72-2D-DSX8/Twall-110.hdf5'
twall_110 = '/share/crsp/lab/amowli/share/BubbleML2/PoolBoiling-WallSuperheat-FC72-2D/Twall-110.hdf5'

twall_100_ds = '/share/crsp/lab/amowli/share/BubbleML2/PoolBoiling-WallSuperheat-FC72-2D-DSX8/Twall-100.hdf5'
twall_100 = '/share/crsp/lab/amowli/share/BubbleML2/PoolBoiling-WallSuperheat-FC72-2D/Twall-100.hdf5'

twall_105_ds = '/share/crsp/lab/amowli/share/BubbleML2/PoolBoiling-WallSuperheat-FC72-2D-DSX8/Twall-105.hdf5'
twall_105 = '/share/crsp/lab/amowli/share/BubbleML2/PoolBoiling-WallSuperheat-FC72-2D/Twall-105.hdf5'

twall_80_ds = '/share/crsp/lab/amowli/share/BubbleML2/PoolBoiling-WallSuperheat-FC72-2D-DSX8/Twall-80.hdf5'
twall_80 = '/share/crsp/lab/amowli/share/BubbleML2/PoolBoiling-WallSuperheat-FC72-2D/Twall-80.hdf5'

twall_60_ds = '/share/crsp/lab/amowli/share/BubbleML2/PoolBoiling-WallSuperheat-FC72-2D-DSX8/Twall-60.hdf5'
twall_60 = '/share/crsp/lab/amowli/share/BubbleML2/PoolBoiling-WallSuperheat-FC72-2D/Twall-60.hdf5'

data_sets = [(twall_60, twall_60_ds), (twall_80, twall_80_ds), 
             (twall_100, twall_100_ds), (twall_105, twall_105_ds), 
             (twall_110, twall_110_ds)]

checkpoint = torch.load(checkpoint_path)

temp_max = checkpoint['max_temp']
vel_max = checkpoint['max_val']

def normalize_data(grid_x, grid_y, temperature, velx, vely, vel_max, temp_max):
    temperature = ((temperature/temp_max)*2)-1
    
    #vel_max = torch.maximum(torch.max(torch.abs(velx)), torch.max(torch.abs(vely)))
    velx = velx/vel_max
    vely = vely/vel_max
    
    grid_x = grid_x/(torch.max(grid_x))
    grid_y = grid_y/(torch.max(grid_y))
    
    nx = grid_x.shape[-2]
    ny = grid_x.shape[-1]
    
    coord = torch.stack([grid_x, grid_y]).reshape([1, 2, nx, ny])

    return coord, temperature, velx, vely

def format_input(temperature, temperature_ds, dfun_ds, velx, vely, coord):
    T = temperature_ds.shape[0]
    
    nx_l = temperature_ds.shape[-2]
    ny_l = temperature_ds.shape[-1]

    nx = temperature.shape[-2]
    ny = temperature.shape[-1]

    batch = ()
    label = ()
    dfun = ()
    for i in range(30, 50):
        temp = torch.Tensor(temperature[i:(i+time_window)]).reshape([1, time_window, nx, ny])
        
        vel_x_p = torch.Tensor(velx[i:(i+time_window)]).reshape([time_window, nx, ny])
        vel_y_p = torch.Tensor(vely[i:(i+time_window)]).reshape([time_window, nx, ny])
        
        vel_p = torch.cat([torch.stack([j, k]) for j, k in zip(vel_x_p, vel_y_p)], dim = 0).reshape([1, time_window*2, nx, ny])
        
        vel_x_f = torch.Tensor(velx[(i+time_window):(i+time_window+future_window)]).reshape([future_window, nx, ny])
        vel_y_f = torch.Tensor(vely[(i+time_window):(i+time_window+future_window)]).reshape([future_window, nx, ny])
        
        vel_f =  torch.cat([torch.stack([j, k]) for j, k in zip(vel_x_f, vel_y_f)], dim = 0).reshape([1, future_window*2, nx, ny])
        
        input = torch.cat((coord, temp, vel_p, vel_f), dim = 1)
        
        label += (temperature_ds[(i+time_window):(i+time_window+future_window)].reshape([1, future_window, nx_l, ny_l]), )
        dfun += (dfun_ds[(i+time_window):(i+time_window+future_window)].reshape([1, future_window, nx_l, ny_l]), )
        batch += (input, )
    
    batch = torch.cat(batch, dim = 0)
    label = torch.cat(label, dim = 0)
    dfun = torch.cat(dfun, dim = 0)

    return batch, label, dfun

random_batch_index = 12

loss_aggregate = []

for i in range(4):
    up_scale = 2**i
    down_scale = 2**(3-i)

    loss_per_layer = []
    
    for layer in range(n_layers):
        output_scaling_factor = [1]*n_layers
        output_scaling_factor[layer] = up_scale

        print("Loading model with upscale of ", up_scale, " on layer ", layer+1)
        print("_____________________________________________________________" )
        
        model = FNO(n_modes=[64, 64],
                            hidden_channels=256,
                            domain_padding=None,
                            in_channels=(time_window*3) + (future_window*2) + 2,
                            out_channels=future_window,
                            n_layers=n_layers,
                            norm='instance_norm',
                            factorization='tucker',
                            implementation='factorized',
                            rank=0.1,
                            output_scaling_factor = output_scaling_factor)

        model.load_state_dict(checkpoint['model_state_dict'])

        loss_per_set = []
        
        for set in data_sets:
            data = h5py.File(set[0], 'r')
            data_ds = h5py.File(set[1], 'r')
            
            coord, temperature, velx, vely = normalize_data(torch.Tensor(data_ds['x'][0]), 
                                                                torch.Tensor(data_ds['y'][0]),
                                                                torch.Tensor(data_ds['temperature']), 
                                                                torch.Tensor(data_ds['velx']), 
                                                                torch.Tensor(data_ds['vely']),
                                                                vel_max,
                                                                temp_max)
            
            temperature_ds = torch.Tensor(data['temperature'][:,::(down_scale),::(down_scale)])
            temperature_ds = ((temperature_ds/temp_max)*2) - 1

            dfun_ds = torch.Tensor(data['dfun'][:,::(down_scale),::(down_scale)])
            
            batch, label, dfun = format_input(temperature, temperature_ds, dfun_ds, velx, vely, coord)

            #print(label.shape, batch.shape)
            with torch.no_grad():
                out = model(batch)
            
            loss = F.mse_loss(out, label)
            loss_interface = interface_rmse(out, label, dfun)

            loss_per_set.append([loss, loss_interface])

            print("Data Set: ", set[0], "\nLosses: \n    MSE: ", loss, "\n    Interface MSE: ", loss_interface, "\n")

            fig, ax = plt.subplots(2, 5, figsize=(15, 5))
    
            for idx in range(5):
                ax[0, idx].imshow(np.flipud(label[random_batch_index, idx].detach().numpy()))
                ax[1, idx].imshow(np.flipud(out[random_batch_index, idx].detach().numpy()))
                ax[0, idx].set_title(f'Step {idx}')
                for row in range(2):
                    ax[row, idx].set_xticks([])
                    ax[row, idx].set_yticks([])
                ax[0,0].set_ylabel('Truth')
                ax[1,0].set_ylabel('Model Out')

            fig.savefig(fname = (re.sub(r'\W+', '', set[0][-14:-5]) + 'scaleX' + str(up_scale) + 'layer#' + str(layer+1)))

            del data, data_ds, coord, temperature, velx, vely, temperature_ds, dfun_ds, batch, label, dfun, out, fig, ax

        del model

        loss_per_layer.append(loss_per_set)
    loss_aggregate.append(loss_per_layer)

loss_aggregate = torch.Tensor(loss_aggregate)

#indexing: scale, layer, set, loss_type

print("Averages: \n")

print("    -per layer:\n")
for i in range(n_layers):
    print("        layer", i , "MSE : ", torch.mean(loss_aggregate[:,i,:,0]), "\nIMSE : ", torch.mean(loss_aggregate[:,i,:,1]), "\n")

print("    -per scale:\n")
for i in range(4):
    print("        scale X", 2**i , "MSE : ", torch.mean(loss_aggregate[i,:,:,0]), "\nIMSE : ", torch.mean(loss_aggregate[i,:,:,1]), "\n")