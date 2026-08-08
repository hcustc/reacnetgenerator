#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later

set -euo pipefail

environment_prefix="${1:?usage: bootstrap-remote-reacnetgenerator.sh ENV_PREFIX}"
conda_executable="/home/chuang/miniconda3/bin/conda"

if [[ ! -x "$environment_prefix/bin/python" ]]; then
    "$conda_executable" create \
        --yes \
        --prefix "$environment_prefix" \
        --override-channels \
        --channel conda-forge \
        python=3.12 \
        reacnetgenerator=1.6.15
fi

"$conda_executable" install \
    --yes \
    --prefix "$environment_prefix" \
    --override-channels \
    --channel conda-forge \
    ase \
    coloredlogs \
    h5py \
    hmmlearn \
    lz4 \
    matplotlib-base \
    networkx \
    numpy \
    openbabel \
    packaging \
    pandas \
    rdkit \
    requests \
    scipy \
    scour \
    tqdm

"$conda_executable" list \
    --prefix "$environment_prefix" \
    --explicit >bootstrap-explicit-spec.txt

"$environment_prefix/bin/python" - <<'PY' | tee bootstrap-report.txt
import importlib.util
import importlib.metadata
import platform
import sys

import ase
import h5py
import numpy
import openbabel
import rdkit
import reacnetgenerator
import scipy
from openbabel import openbabel as ob
from reacnetgenerator import dps

print(f"hostname={platform.node()}")
print(f"python={sys.executable}")
print(f"python_version={sys.version.split()[0]}")
print(f"reacnetgenerator_version={reacnetgenerator.__version__}")
print(
    "reacnetgenerator_distribution="
    f"{importlib.metadata.version('reacnetgenerator')}"
)
print(f"reacnetgenerator_module={reacnetgenerator.__file__}")
print(f"dps_module={dps.__file__}")
print(f"numpy={numpy.__version__}")
print(f"scipy={scipy.__version__}")
print(f"h5py={h5py.__version__}")
print(f"ase={ase.__version__}")
print(f"rdkit={rdkit.__version__}")
print(f"openbabel={openbabel.__file__}")
print(f"openbabel_release={ob.OBReleaseVersion()}")
print(f"dps_spec={importlib.util.find_spec('reacnetgenerator.dps').origin}")
PY
