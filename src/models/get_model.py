import os
from neuralop.models import FNO, UNO
from .factorized_fno.factorized_fno import FNOFactorized2DBlock 
from .gefno.gfno import GFNO2d
from .pdebench.unet import UNet2d 
from .pdearena.unet import Unet, FourierUnet

from torch.nn.parallel import DistributedDataParallel as DDP


_UNET2D = 'unet2d'

_UNET_MOD_ATTN = 'unet_mod_attn'
_UFNET = 'ufnet'

_FNO = 'fno'
_UNO = 'uno'

_FFNO = 'factorized_fno'

_GFNO = 'gfno'

_MODEL_LIST = [
    _UNET2D,
    _UNET_MOD_ATTN,
    _UFNET,
    _FNO,
    _UNO,
    _FFNO,
    _GFNO
]

def get_model(model_name, in_channels, out_channels, exp):
    assert model_name in _MODEL_LIST, f'Model name {model_name} invalid'
    if model_name == _UNET_MOD_ATTN:
        model = Unet(in_channels=in_channels,
                     out_channels=out_channels,
                     hidden_channels=exp.model.hidden_channels,
                     activation='gelu',
                     mid_attn=False,
                     norm=True,
                     use1x1=True)
    elif model_name == _UNET2D: 
        model = UNet2d(in_channels=in_channels,
                       out_channels=out_channels,
                       init_features=exp.model.init_features)
    elif model_name == _UFNET:
        model = FourierUnet(in_channels=in_channels,
                            out_channels=out_channels,
                            hidden_channels=exp.model.hidden_channels,
                            modes1=exp.model.modes1,
                            modes2=exp.model.modes2,
                            norm=True,
                            n_fourier_layers=exp.model.n_fourier_layers)
    elif model_name == _FNO:
        model = FNO(n_modes=exp.model.n_modes,
                    hidden_channels=exp.model.hidden_channels,
                    domain_padding=exp.model.domain_padding,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_layers=exp.model.n_layers,
                    factorization='tucker',
                    implementation='factorized',
                    rank=0.05)
    elif model_name == _UNO:
        model = UNO(in_channels=in_channels, 
                    out_channels=out_channels,
                    hidden_channels=exp.model.hidden_channels,
                    projection_channels=exp.model.projection_channels,
                    uno_out_channels=[32,64,64,64,32],
                    uno_n_modes=[[32,32],[16,16],[16,16],[16,16],[32,32]],
                    uno_scalings=[[1,1],[0.5,0.5],[1,1],[1,1],[2,2]],
                    n_layers=exp.model.n_layers,
                    domain_padding=exp.model.domain_padding)
    elif model_name == _FFNO:
        model = FNOFactorized2DBlock(in_channels=in_channels,
                                     out_channels=out_channels,
                                     modes=exp.model.modes,
                                     width=exp.model.width,
                                     dropout=exp.model.dropout,
                                     n_layers=exp.model.n_layers,
                                     )
    elif model_name == _GFNO:
        model = GFNO2d(in_channels=in_channels,
                       out_channels=out_channels,
                       modes=exp.model.modes,
                       width=exp.model.width,
                       reflection=exp.model.reflection) 
    if exp.distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        model = model.to(local_rank).float()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
    else:
        model = model.cuda().float()
    return model
