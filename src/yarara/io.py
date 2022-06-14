"""
This modules does XXX
"""

import os
import pickle
from io import BufferedWriter
from typing import Any, Optional, Union

import pandas as pd
from astropy.io import fits
from numpy import str_

from . import pickle_protocol_version


def touch_pickle(filename):
    if not os.path.exists(filename):
        pickle_dump({},open(filename,'wb'))
        return {}
    else:
        return pd.read_pickle(filename)

def open_pickle(filename):
    if filename.split(".")[-1] == "p":
        a = pd.read_pickle(filename)
        return a
    elif filename.split(".")[-1] == "fits":
        data = fits.getdata(filename)
        header = fits.getheader(filename)
        return data, header


def save_pickle(filename: Union[str_, str], output: Any, header: None = None) -> None:
    if filename.split(".")[-1] == "p":
        pickle.dump(output, open(filename, "wb"), protocol=pickle_protocol_version)
    if filename.split(".")[-1] == "fits":  # for futur work
        pass


def pickle_dump(obj: Any, obj_file: BufferedWriter, protocol: Optional[int] = None) -> None:
    if protocol is None:
        protocol = pickle_protocol_version
    pickle.dump(obj, obj_file, protocol=protocol)
