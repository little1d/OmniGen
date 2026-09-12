from .base import BaseDataLoader
from .registry import register_loader, create_loader_for_source, list_loaders
from .data_structures import DataSample, Dataset
from .sdf_caption_loader import SDFCaptionLoader

try:
    from .spectrum_loader import SpectrumDataLoader
except ModuleNotFoundError as error:
    if error.name != "datasets":
        raise
    SpectrumDataLoader = None

# Register loaders (already registered via decorators in the class definition)
# No need to manually register - @register_loader decorator handles it

__all__ = [
    'BaseDataLoader',
    'SDFCaptionLoader',
    'SpectrumDataLoader',
    'register_loader',
    'create_loader_for_source',
    'DataSample',
    'Dataset',
    'list_loaders',
]
