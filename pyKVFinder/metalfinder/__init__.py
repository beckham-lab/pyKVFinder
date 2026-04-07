"""
Metal Binding Site Identification Module

This module extends pyKVFinder to identify potential metal binding sites
in protein structures by converting cavity detection outputs into filtered
probe positions suitable for metal ion placement.
"""

__version__ = "1.0.0"

from .probe_converter import ProbeSet, ProbeConverter
from .filters import (
    BaseFilter,
    FilterResult,
    DistanceFilter,
    CoordinationFilter,
    HardCoordinationFilter,
    SphereDonorFilter,
    SignatureDeduplicator,
    CoordinationInfo,
    CoordinationSignature,
    run_filter_pipeline
)
from .pdb_parser import parse_pdb

__all__ = [
    # Probe conversion
    'ProbeSet',
    'ProbeConverter',
    # Filters
    'BaseFilter',
    'FilterResult',
    'DistanceFilter',
    'CoordinationFilter',
    'HardCoordinationFilter',
    'SphereDonorFilter',
    'SignatureDeduplicator',
    'CoordinationInfo',
    'CoordinationSignature',
    'run_filter_pipeline',
    # Utilities
    'parse_pdb',
]
