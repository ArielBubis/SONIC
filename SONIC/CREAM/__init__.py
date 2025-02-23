"""
SONIC CREAM Module

This module provides various utilities and functionalities for the SONIC project,
including data conversion, dataset handling, utility functions, I/O operations, 
and data splitting.

Modules:
    - convert: Functions for converting data formats.
    - dataset: Classes and functions for handling datasets.
    - sonic_utils: Utility functions for various tasks.
    - sonic_io: Functions for input/output operations.
    - split: Functions for splitting data into training, validation, and test sets.
"""
from . import convert as convert
from . import dataset as dataset
from . import sonic_utils as utils
from . import sonic_io as io
from . import split as split