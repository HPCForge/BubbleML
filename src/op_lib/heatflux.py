import torch

def denormalize_temp_grad(temp, t_wall, t_bulk=58, k=0.054):
    print('t_wall: ', t_wall)
    del_t = t_wall - t_bulk
    return 2 * k * del_t * (1 - temp)

def heatflux(temp, dfun, t_wall, x, dy):
    r"""
    heat flux, q=dT/dy, is the temperature in the liquid phase in cells directly
    above the heater. 
    temp and dfun are layed out T x row x col
    dy is the grid spacing in the y direction. 
    """
    assert temp.dim() == 3
    assert temp.size() == dfun.size()
    assert temp.size() == x.size()
    lc = 0.0007

    d_temp = denormalize_temp_grad(temp[:, 0], t_wall)
    heater_mask = (x >= -2.5) & (x <= 2.5)
    liquid_mask = dfun < 0
    hflux_list = torch.mean((heater_mask[:, 0] & liquid_mask[:, 0]).to(float) * 
                                d_temp / (dy * lc),
                            dim=1)

    hflux = torch.mean(hflux_list)
    qmax = torch.max(hflux_list)
    return hflux, qmax
