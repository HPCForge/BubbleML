import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

class Temp_PDE_Loss(object):
    def __init__(self, runtime_params, decay_rate):
        
        self.param_dict = {}
        for param in runtime_params:
            self.param_dict[param[0].decode().strip()] = param[1]
            
        self.decay_rate = decay_rate
    
    def compute_gradient(self, temp_field, resolution):
        # N = temp_field.shape[-len(resolution):]
            
        # assert len(resolution) == len(N), "Length of resolution must correspond to dimension of temprature field."
        grad = []
        
        for (i,res) in enumerate(resolution):
            dim = [-len(resolution)+((i+k)%len(resolution)) for k in range(1,len(resolution),1)]
            grad.append(self.trim_ends(self.compute_derivative(temp_field, res, -len(resolution)+i), dim).unsqueeze(0))
    
        grad = torch.cat(grad, dim = 0)
        
        return grad
        # k = []
       
        # for (i, n) in enumerate(N):
        #     reps = [1]+list(N)
        #     resh = [1] * (len(N)+1)
        #     resh[i+1] = n
        #     reps[i+1] = 1
        #     k.append(torch.fft.fftfreq(n, resolution[i]).reshape(resh).repeat(reps)*resolution[i])
    
        # k = torch.cat(k, dim = 0)
    
        # reps = [1] * (len(N)+1)
        # reps[0] = len(N)
        # dim = list(range(-len(N), 0, 1))
        
        # temp_lap = torch.fft.fftn(temp_field, dim = dim).unsqueeze(0).repeat(reps)
        # temp_lap = temp_lap * (torch.exp(2j * np.pi * k) - 1)
        # temp_lap = torch.fft.irfftn(temp_lap[..., :N[-1]//2 + 1], dim = dim)  # Compute the inverse FFT
        
        # for (i, res) in enumerate(resolution):
        #     temp_lap[i] = temp_lap[i]/res
        #     temp_lap = torch.narrow(temp_lap, -(i+1), 0, N[-(i+1)]-1) 
        
        # return temp_lap
    
    def compute_derivative(self, field, resolution, dim):
        N = field.shape
        
        resh = [1] * len(N)
        resh[dim] = N[dim]
    
        reps = list(N)
        reps[dim] = 1
        
        k = torch.fft.fftfreq(N[dim], resolution).reshape(resh).repeat(reps)*resolution
    
        field_derivative = torch.fft.fft(field, dim = dim)
        field_derivative = field_derivative * (torch.exp(2j * np.pi * k) - 1)
        field_derivative = torch.narrow(field_derivative, dim, 0, N[dim]//2 + 1)
        field_derivative = torch.fft.irfft(field_derivative, dim = dim)
        field_derivative = torch.narrow(field_derivative, dim, 0, N[dim]-1)/resolution
    
        return field_derivative
    
    def compose_diffusivity(self, dfun):        
        Re = self.param_dict['ins_invreynolds']
        Pr = self.param_dict['ht_prandtl']
        k = self.param_dict['mph_thcogas']
        Cp = self.param_dict['mph_cpgas']
        rho = self.param_dict['mph_rhogas']
        
        alpha = k/(Cp*rho)
    
        diffusivity = torch.ones(dfun.shape)
        diffusivity *= Re*(1/Pr)
    
        diffusivity = torch.where(dfun > 0, diffusivity*alpha, diffusivity)
    
        return diffusivity
    
    def heat_equation2D(self, temp_field, velx, vely, dfun, resolution):
        grad_1 = self.compute_gradient(temp_field, resolution)
    
        temporal_grad = grad_1[0]
        spatial_grad = grad_1[1:]
    
        diffu = self.trim_ends(self.compose_diffusivity(dfun), [-2, -1])
    
        reps = [1]*(len(diffu.shape)+1)
        reps[0] = 2
        diffu.unsqueeze(0).repeat(reps)
    
        spatial_grad_scaled = diffu*spatial_grad
    
        spatial_grad_scaled = F.pad(spatial_grad_scaled, (0,1,0,1), 'constant', 0)
        #spatial_grad_scaled[-2] = F.pad(spatial_grad_scaled[-2], (0,0,0,1), 'constant', 0)
    
        grad_2_x = self.compute_derivative(spatial_grad_scaled[-1], resolution[-1], -1)
        grad_2_y = self.compute_derivative(spatial_grad_scaled[-2], resolution[-2], -2)
    
        grad_2_x = self.trim_ends(self.trim_ends(grad_2_x, [-2, -1]), [-2])
        grad_2_y = self.trim_ends(self.trim_ends(grad_2_y, [-2, -1]), [-1])
    
        vel = torch.stack((self.trim_ends(vely, [-2, -1]), self.trim_ends(velx, [-2, -1])))
        convection = vel*spatial_grad
        convection = torch.sum(convection, dim = 0)
    
        convection = self.trim_ends(convection, [-2, -1])
    
        temporal_grad = self.trim_ends(temporal_grad, [-2, -1])
    
        return temporal_grad, convection, grad_2_x, grad_2_y, spatial_grad
    
    def heat_loss_function(self, temp_field, velx, vely, dfun, resolution):
        temporal_grad, convection, grad_2_x, grad_2_y, spatial_grad = self.heat_equation2D(temp_field, velx, vely, dfun, resolution)
        
        return temporal_grad+convection-grad_2_x-grad_2_y
    
    def trim_ends(self, field, dim):
        for d in dim:
            field = torch.narrow(field, d, 0, field.shape[d] - 1)
    
        return field
    
    def get_interface_mask(self, dgrid):
        [batch, window, rows, cols] = dgrid.shape
        up, down= dgrid[:,:,1:,:]*dgrid[:,:,:-1,:], dgrid[:,:,1:,:]*dgrid[:,:,:-1,:]
        left, right = dgrid[:,:,:,1:]*dgrid[:,:,:,:-1], dgrid[:,:,:,1:]*dgrid[:,:,:,:-1]
    
        side_pad = torch.ones(batch, window, rows, 1)
        top_pad = torch.ones(batch, window, 1, cols)
    
        left = torch.cat((side_pad, left), dim = -1)
        right = torch.cat((right, side_pad), dim = -1)
        down = torch.cat((top_pad, down), dim = -2)
        up = torch.cat((up, top_pad), dim = -2)
    
        mask = ((left < 0) + (right < 0) + (up < 0) + (down < 0)) > 0
        return mask
    
    def temporal_causality_decay_factor(self, heat_loss, dfun):
        mask = self.get_interface_mask(dfun)
        mask = self.trim_ends(self.trim_ends(mask, [-2,-1]), [-2,-1])
        envelope = [1]
        for i in range(heat_loss.shape[-3]-1):
            envelope.append(torch.exp((-self.decay_rate)*torch.mean(heat_loss[...,:(i+1),:,:][mask[...,:(i+1),:,:]]**2)))
    
        return torch.Tensor(envelope)

    def __call__(self, temp_field, velx, vely, dfun, resolution):
        heat_loss = self.heat_loss_function(temp_field, velx, vely, dfun, resolution) 
        TCDF = self.temporal_causality_decay_factor(heat_loss, dfun)
        heat_loss = torch.square(heat_loss)
        for (i, factor) in enumerate(TCDF):
            heat_loss[:, i] * factor
        
        return torch.mean(heat_loss)
