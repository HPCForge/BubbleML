import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

class Temp_PDE_Loss(object):
    def __init__(self, runtime_params, decay_rate):
        
        self.param_dict = runtime_params
        self.decay_rate = decay_rate
    
    def compute_gradient(self, temp_field, resolution):
        # N = temp_field.shape[-len(resolution):]
            
        # assert len(resolution) == len(N), "Length of resolution must correspond to dimension of temprature field."
        grad = []
        
        for (i,res) in enumerate(resolution):
            dim = [-len(resolution)+((i+k)%len(resolution)) for k in range(1,len(resolution),1)]
            if i == 0:
                grad.append(self.trim_ends(self.compute_derivative(temp_field, res, -len(resolution)+i, -1), dim).unsqueeze(0))
            else:
                grad.append(self.trim_ends(self.compute_derivative(temp_field, res, -len(resolution)+i, 1), dim).unsqueeze(0))
    
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
    
    def compute_derivative(self, field, resolution, dim, direction = 1):
        N = field.shape
        
        resh = [1] * len(N)
        resh[dim] = N[dim]
    
        reps = list(N)
        reps[dim] = 1
        
        k = (torch.fft.fftfreq(N[dim], resolution).reshape(resh).repeat(reps)*resolution).cuda()
    
        field_derivative = torch.fft.fft(field, dim = dim)

        if direction == 1:
            field_derivative = field_derivative * (torch.exp(2j * np.pi * k) - 1)
        else:
            field_derivative = field_derivative * (1 - torch.exp(-2j * np.pi * k))
        
        field_derivative = torch.narrow(field_derivative, dim, 0, N[dim]//2 + 1)
        field_derivative = torch.fft.irfft(field_derivative, dim = dim)

        if direction == 1:
            field_derivative = torch.narrow(field_derivative, dim, 0, N[dim]-1)/resolution
        else:
            field_derivative = torch.narrow(field_derivative, dim, 1, N[dim]-1)/resolution
    
        return field_derivative
    
    def compose_diffusivity(self, dfun):        
        Re = self.param_dict['ins_invreynolds'][0]
        Pr = self.param_dict['ht_prandtl'][0]
        k = self.param_dict['mph_thcogas'][0]
        Cp = self.param_dict['mph_cpgas'][0]
        rho = self.param_dict['mph_rhogas'][0]

        #print("RUNTIMEPARAMS: ", Re, Pr, k, Cp, rho)
        
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
    
        spatial_grad_scaled = diffu.cuda()*spatial_grad
    
        spatial_grad_scaled = F.pad(spatial_grad_scaled, (0,1,0,1), 'constant', 0)
        #spatial_grad_scaled[-2] = F.pad(spatial_grad_scaled[-2], (0,0,0,1), 'constant', 0)
    
        grad_2_x = self.compute_derivative(spatial_grad_scaled[-1], resolution[-1], -1)
        grad_2_y = self.compute_derivative(spatial_grad_scaled[-2], resolution[-2], -2)
    
        grad_2_x = self.trim_ends(self.trim_ends(grad_2_x, [-2, -1]), [-2])
        grad_2_y = self.trim_ends(self.trim_ends(grad_2_y, [-2, -1]), [-1])
    
        vel = torch.stack((self.trim_ends(vely, [-2, -1]), self.trim_ends(velx, [-2, -1])))
        convection = vel.cuda()*spatial_grad
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
    
    def trim_start(self, field, dim):
        for d in dim:
            field = torch.narrow(field, d, 1, field.shape[d])
    
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
        for (i, factor) in enumerate(TCdfun):
            heat_loss[:, i] * factor
        
        return torch.mean(heat_loss)


class Vel_PDE_Loss(object):
    def __init__(self, runtime_params, decay_rate):
        
        self.param_dict = runtime_params
        self.decay_rate = decay_rate
    
    def compute_gradient(self, vel, resolution):
        # N = temp_field.shape[-len(resolution):]
            
        # assert len(resolution) == len(N), "Length of resolution must correspond to dimension of temprature field."
        grad = []
        
        for (i,res) in enumerate(resolution):
            dim = [-len(resolution)+((i+k)%len(resolution)) for k in range(1,len(resolution),1)]
            if i == 0:
                grad.append(self.trim_ends(self.compute_derivative(vel, res, -len(resolution)+i, -1), dim).unsqueeze(0))
            else:
                grad.append(self.trim_ends(self.compute_derivative(vel, res, -len(resolution)+i, 1), dim).unsqueeze(0))
    
        for i in grad:
            print(i.shape)
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
    
    def compute_derivative(self, field, resolution, dim, direction = 1):
        N = field.shape
        
        resh = [1] * len(N)
        resh[dim] = N[dim]
    
        reps = list(N)
        reps[dim] = 1
        
        k = (torch.fft.fftfreq(N[dim], resolution).reshape(resh).repeat(reps)*resolution).cuda()
    
        field_derivative = torch.fft.fft(field, dim = dim)

        if direction == 1:
            field_derivative = field_derivative * (torch.exp(2j * np.pi * k) - 1)
        else:
            field_derivative = field_derivative * (1 - torch.exp(-2j * np.pi * k))
        
        field_derivative = torch.narrow(field_derivative, dim, 0, N[dim]//2 + 1)
        field_derivative = torch.fft.irfft(field_derivative, dim = dim)

        if direction == 1:
            field_derivative = torch.narrow(field_derivative, dim, 0, N[dim]-1)/resolution
        else:
            field_derivative = torch.narrow(field_derivative, dim, 1, N[dim]-1)/resolution
    
        return field_derivative
    
    def compose_viscosity(self, dfun):        
        inv_Re = self.param_dict['ins_invreynolds'][0]
        rho = self.param_dict['mph_rhogas'][0]
        mu = self.param_dict['mph_mugas'][0]
            
        viscosity = torch.ones(dfun.shape)
        viscosity *= inv_Re
    
        viscosity = torch.where(dfun > 0, viscosity*(mu/rho), viscosity)
    
        return viscosity
    
    def compose_pressure(self, dfun):
        rho = self.param_dict['mph_rhogas'][0]
            
        pressure = torch.ones(dfun.shape)
        pressure = torch.where(dfun > 0, pressure*(1/rho), pressure)
    
        return pressure






    def velocity_equation2D(self, pressure_field, velx, vely, dfun, resolution):
        
        combined_magnitude = torch.sqrt(velx**2 + vely**2)
        grad_mag = self.compute_gradient(combined_magnitude, resolution)
        combined_magnitude = self.trim_ends(combined_magnitude, [-3, -2, -1])

        du_dt = grad_mag[0]
        grad_vel = grad_mag[1:]

        u_grad_u = grad_vel * combined_magnitude


        grad_pressure = self.compute_gradient(pressure_field, resolution)[1:]

        pres_const = self.compose_pressure(dfun)
        reps = [1]*(len(pres_const.shape)+1)
        reps[0] = 2
        pres_const.unsqueeze(0).repeat(reps)
        pres_const = self.trim_ends(pres_const, [-2, -1])
        pres_const = self.trim_start(pres_const, [-3])

        pressure_term =  pres_const.cuda() * grad_pressure

        visc_const = self.compose_viscosity(dfun)
        visc_const = self.trim_ends(visc_const, [-2, -1])
        visc_const = self.trim_start(visc_const, [-3])
        spatial_grad_scaled = visc_const.cuda()*grad_vel

        grad_2_x = self.compute_derivative(spatial_grad_scaled[-1], resolution[-1], -1)
        grad_2_y = self.compute_derivative(spatial_grad_scaled[-2], resolution[-2], -2)
        grad_2_x = self.trim_ends(grad_2_x, [-2])
        grad_2_y = self.trim_ends(grad_2_y, [-1])

        # Match size of grad_2
        pressure_term = self.trim_ends(pressure_term, [-2, -1])
        u_grad_u = self.trim_ends(u_grad_u, [-2. -1])
        du_dt = self.trim_ends(du_dt, [-2. -1])
        return du_dt, u_grad_u, pressure_term, grad_2_x, grad_2_y
    
    def vel_loss_function(self, pressure_field, velx, vely, dfun, resolution):
        temporal_grad, convection, grad_pressure, grad_2_x, grad_2_y = self.velocity_equation2D(pressure_field, velx, vely, dfun, resolution)
        
        return temporal_grad+convection+grad_pressure-grad_2_x-grad_2_y - 1j
    
    def trim_ends(self, field, dim):
        for d in dim:
            field = torch.narrow(field, d, 0, field.shape[d] - 1)
    
        return field
    
    def trim_start(self, field, dim):
        for d in dim:
            field = torch.narrow(field, d, 1, field.shape[d]-1)
    
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
    
    def temporal_causality_decay_factor(self, vel_loss, dfun):
        mask = self.get_interface_mask(dfun)
        mask = self.trim_ends(self.trim_ends(mask, [-2,-1]), [-2,-1])
        envelope = [1]
        for i in range(vel_loss.shape[-3]-1):
            envelope.append(torch.exp((-self.decay_rate)*torch.mean(vel_loss[...,:(i+1),:,:][mask[...,:(i+1),:,:]]**2)))
    
        return torch.Tensor(envelope)

    def __call__(self, pressure_field, velx, vely, dfun, resolution):
        vel_loss = self.vel_loss_function(pressure_field, velx, vely, dfun, resolution) 
        TCDF = self.temporal_causality_decay_factor(vel_loss, dfun)
        vel_loss = torch.square(vel_loss)
        for (i, factor) in enumerate(TCdfun):
            vel_loss[:, i] * factor
        
        return torch.mean(vel_loss)