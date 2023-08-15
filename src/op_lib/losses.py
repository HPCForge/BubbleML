r"""
relative LpLoss. Implementation taken and modified from
https://github.com/neuraloperator/neuraloperator
"""
import math
import torch
import torch.nn.functional as F

class LpLoss(object):
    def __init__(self, d=1, p=2, L=2*math.pi, reduce_dims=0, reductions='sum', add_PDE_LOSS = False):
        super().__init__()

        self.d = d
        self.p = p

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims
        
        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean'
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean'
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L]*self.d
        else:
            self.L = L
    
    def uniform_h(self, x):
        h = [0.0]*self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.size(-j)
        
        return h

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        
        return x

    def rel(self, x, y):
        diff = torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                          p=self.p, dim=-1, keepdim=False)
        ynorm = torch.norm(torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)

        diff = diff/ynorm

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, x, y):
        return self.rel(x, y)


class H1Loss(object):
    def __init__(self, d=1, L=2*math.pi, reduce_dims=0, reductions='sum', fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
        super().__init__()

        assert d > 0 and d < 4, "Currently only implemented for 1, 2, and 3-D."

        self.d = d
        self.fix_x_bnd = fix_x_bnd
        self.fix_y_bnd = fix_y_bnd
        self.fix_z_bnd = fix_z_bnd

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean'
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean'
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L]*self.d
        else:
            self.L = L

    def compute_terms(self, x, y, h):
        dict_x = {}
        dict_y = {}

        if self.d == 1:
            dict_x[0] = x
            dict_y[0] = y

            x_x = central_diff_1d(x, h[0], fix_x_bnd=self.fix_x_bnd)
            y_x = central_diff_1d(y, h[0], fix_x_bnd=self.fix_x_bnd)

            dict_x[1] = x_x
            dict_y[1] = y_x

        elif self.d == 2:
            dict_x[0] = torch.flatten(x, start_dim=-2)
            dict_y[0] = torch.flatten(y, start_dim=-2)

            x_x, x_y = central_diff_2d(x, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd)
            y_x, y_y = central_diff_2d(y, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd)

            dict_x[1] = torch.flatten(x_x, start_dim=-2)
            dict_x[2] = torch.flatten(x_y, start_dim=-2)

            dict_y[1] = torch.flatten(y_x, start_dim=-2)
            dict_y[2] = torch.flatten(y_y, start_dim=-2)

        else:
            dict_x[0] = torch.flatten(x, start_dim=-3)
            dict_y[0] = torch.flatten(y, start_dim=-3)

            x_x, x_y, x_z = central_diff_3d(x, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)
            y_x, y_y, y_z = central_diff_3d(y, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)

            dict_x[1] = torch.flatten(x_x, start_dim=-3)
            dict_x[2] = torch.flatten(x_y, start_dim=-3)
            dict_x[3] = torch.flatten(x_z, start_dim=-3)

            dict_y[1] = torch.flatten(y_x, start_dim=-3)
            dict_y[2] = torch.flatten(y_y, start_dim=-3)
            dict_y[3] = torch.flatten(y_z, start_dim=-3)

        return dict_x, dict_y

    def uniform_h(self, x):
        h = [0.0]*self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.size(-j)

        return h

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)

        return x

    def rel(self, x, y, h=None):
        #Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d

        dict_x, dict_y = self.compute_terms(x, y, h)

        diff = torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2
        ynorm = torch.norm(dict_y[0], p=2, dim=-1, keepdim=False)**2

        for j in range(1, self.d + 1):
            diff += torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False)**2
            ynorm += torch.norm(dict_y[j], p=2, dim=-1, keepdim=False)**2

        diff = (diff**0.5)/(ynorm**0.5)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff


    def __call__(self, x, y, h=None):
        return self.rel(x, y, h=h)

def temp_stokes_loss(T, u, v):
    batchsize = T.size(0)
    nx = T.size(2)
    ny = T.size(3)
    
    device = T.device
    T = T.reshape(batchsize, nx, ny)
    u = u.reshape(batchsize, nx, ny)
    v = v.reshape(batchsize, nx, ny)

    T_h = torch.fft.fft2(T, dim=[-2, -1])
    u_h = torch.fft.fft2(u, dim=[-2, -1])
    v_h = torch.fft.fft2(v, dim=[-2, -1])
    # Wavenumbers in y-direction
    k_maxx = nx//2
    k_maxy = ny//2
    Nx = nx
    Ny = ny
    k_x = torch.cat((torch.arange(start=0, end=k_maxx, step=1, device=device),
                     torch.arange(start=-k_maxx, end=0, step=1, device=device)), 0).reshape(Nx, 1).repeat(1, Ny).reshape(1,Nx,Ny)
    k_y = torch.cat((torch.arange(start=0, end=k_maxy, step=1, device=device),
                     torch.arange(start=-k_maxy, end=0, step=1, device=device)), 0).reshape(1, Ny).repeat(Nx, 1).reshape(1,Nx,Ny)
    #Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0] = 1.0

    Ty_h = 1j * k_y * T_h
    Tx_h = 1j * k_x * T_h
    Tlap_h = lap * T_h

    Txu_conv_h = F.conv2d(u_h, Tx_h, stride = 1, padding = (0,0))*(1/(4*(math.pi**2)))
    Tyv_conv_h = F.conv2d(v_h, Ty_h, stride = 1, padding = (0,0))*(1/(4*(math.pi**2)))

    gradTdotu_h = Tyv_conv_h + Txu_conv_h

    gradTdotu = torch.fft.irfft2(gradTdotu_h[:, :, :k_maxy + 1], dim=[-2, -1])
    Tlap = torch.fft.irfft2(Tlap_h[:, :, :k_maxy+1], dim=[-2,-1])

    PDE_LOSS = torch.sum(torch.square(gradTdotu - Tlap))
    return PDE_LOSS
