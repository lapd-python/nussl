#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .masks import *
from .separation_base import SeparationBase
from .mask_separation_base import MaskSeparationBase

# Repetition/median based methods
from .repet import Repet
from .repet_sim import RepetSim
from .ft2d import FT2D
from .hpss import HPSS

# Melody-based methods
from .melodia import Melodia

# Spatialization based methods
from .duet import Duet
from .projet import Projet

# Others
from .nmf_mfcc import NMF_MFCC
from .ideal_mask import IdealMask
from .overlap_add import OverlapAdd
from .ica import ICA
from .high_low_pass_filter import HighLowPassFilter
from .deep_clustering import DeepClustering

__all__ = ['SeparationBase', 'MaskSeparationBase',
           'Repet', 'RepetSim', 'FT2D', 'Duet', 'Projet', 'Melodia','IdealMask', 'OverlapAdd',
           'ICA', 'HighLowPassFilter', 'NMF_MFCC', 'DeepClustering', 'HPSS']
