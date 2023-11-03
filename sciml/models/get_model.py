import os
from neuralop.models import FNO, UNO
from .factorized_fno.factorized_fno import FNOFactorized2DBlock 
from .gefno.gfno import GFNO2d
from .gefno.gcnn import GCNN2d
from .pdebench.unet import UNet2d 
from .pdearena.unet import Unet, FourierUnet

from torch.nn.parallel import DistributedDataParallel as DDP


_UNET_BENCH = 'unet_bench'

_UNET_ARENA = 'unet_arena'
_UFNET = 'ufnet'

_FNO = 'fno'
_UNO = 'uno'

_FFNO = 'factorized_fno'

_GFNO = 'gfno'
_GCNN = 'gcnn'

_MODEL_LIST = [
    _UNET_BENCH,
    _UNET_ARENA,
    _UFNET,
    _FNO,
    _UNO,
    _FFNO,
    _GFNO,
    _GCNN
]

_FOURIER_MODELS = [
    _FNO,
    _FFNO,
    #_GFNO,
]

def get_model(model_name,
              in_channels,
              out_channels,
              domain_rows,
              domain_cols,
              exp):
    # FNO people recommended using n_modes around 2/3 resolution.
    # Since the flow boiling datasets are so wide, that may not be
    # possible along the x-direction.
    fmode_row, fmode_col = None, None
    if model_name in _FOURIER_MODELS:
        fmode_row = int(exp.model.fmode_frac[0] * domain_rows)
        fmode_col = int(exp.model.fmode_frac[1] * domain_cols)

    assert model_name in _MODEL_LIST, f'Model name {model_name} invalid'
    if model_name == _UNET_ARENA:
        model = Unet(in_channels=in_channels,
                     out_channels=out_channels,
                     hidden_channels=exp.model.hidden_channels,
                     ch_mults=[1,2,2,4,4],
                     is_attn=[False]*5,
                     activation='gelu',
                     mid_attn=False,
                     norm=True,
                     use1x1=True)
    elif model_name == _UNET_BENCH: 
        model = UNet2d(in_channels=in_channels,
                       out_channels=out_channels,
                       init_features=exp.model.init_features)
    elif model_name == _UFNET:
        model = FourierUnet(in_channels=in_channels,
                            out_channels=out_channels,
                            hidden_channels=exp.model.hidden_channels,
                            # UFNET's fourier layers are in the middle of
                            # the U, so it doesn't make sense to use the 2/3
                            # setting like we do for the other models.
                            modes1=exp.model.modes1,
                            modes2=exp.model.modes2,
                            norm=True,
                            n_fourier_layers=exp.model.n_fourier_layers)
    elif model_name == _FNO:
        model = FNO(n_modes=(fmode_row, fmode_col),
                    hidden_channels=exp.model.hidden_channels,
                    domain_padding=exp.model.domain_padding,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_layers=exp.model.n_layers,
                    norm=exp.model.norm,
                    separable=exp.model.separable)
    elif model_name == _UNO:
        model = UNO(in_channels=in_channels, 
                    out_channels=out_channels,
                    hidden_channels=exp.model.hidden_channels,
                    projection_channels=exp.model.projection_channels,
                    uno_out_channels=exp.model.uno_out_channels,
                    uno_n_modes=exp.model.uno_n_modes,
                    uno_scalings=exp.model.uno_scalings,
                    n_layers=exp.model.n_layers,
                    domain_padding=exp.model.domain_padding)
    elif model_name == _FFNO:
        model = FNOFactorized2DBlock(in_channels=in_channels,
                                     out_channels=out_channels,
                                     # FFNO modes need to be halved. Unlike neuralop,
                                     # it does not have it for us.
                                     modes=64,
                                     width=exp.model.width,
                                     dropout=exp.model.dropout,
                                     n_layers=exp.model.n_layers,
                                     )
    elif model_name == _GFNO:
        model = GFNO2d(in_channels=in_channels,
                       out_channels=out_channels,
                       modes=exp.model.modes, #fmode_row,
                       width=exp.model.width,
                       reflection=exp.model.reflection) 
    elif model_name == _GCNN:
        model = GCNN2d(in_channels=in_channels,
                       out_channels=out_channels,
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
