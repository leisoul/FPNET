import os.path as osp
import importlib
from os import scandir

model_folder = osp.dirname(osp.abspath(__file__))

model_filenames = [
    osp.splitext(osp.basename(v.path))[0] for v in scandir(model_folder)
    if v.name.endswith('.py')
]

_model_modules = [
    importlib.import_module(f'models.{file_name}')
    for file_name in model_filenames
]

_loss = [
    importlib.import_module(f'models.losses')
]


def create_model(opt, logger):
    model_type = opt['network_g']['type']
    model_cls = None
    for module in _model_modules:
        model_cls = getattr(module, model_type, None)
        if model_cls is not None:
            break
    if model_cls is None:
        raise ValueError(f'Model {model_type} is not found.')

    model_params = {k: v for k, v in opt['network_g'].items() if k != 'type'}

    logger.info(f'Model parameters: {model_params}')   
     
    model = model_cls(**model_params)

    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model


def create_loss(opt, logger):
    loss_type = opt['train']['pixel_opt']['type']
    loss_cls = None

    for loss in _loss:
        loss_cls = getattr(loss, loss_type, None)
        if loss_cls is not None:
            break
    if loss_cls is None:
        raise ValueError(f'Loss {loss_cls} is not found.')

    loss = loss_cls()

    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
