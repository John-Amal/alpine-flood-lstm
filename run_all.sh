#!/usr/bin/env bash
# One-shot pipeline: prepare -> train -> evaluate -> analyse.
# Usage: ./run_all.sh [fast|full] [cpu|mps|cuda:0]
set -euo pipefail
VARIANT=${1:-fast}
DEVICE=${2:-cpu}
DATA=${DATA:-data/Caravan_extension_CH}

python scripts/prepare.py --data-dir "$DATA" --device "$DEVICE"
nh-run train --config-file "configs/${VARIANT}.yml"
RUN=$(ls -td runs/camelsch_regional_lstm_${VARIANT}_* | head -1)
nh-run evaluate --run-dir "$RUN"
python scripts/analyse.py --run-dir "$RUN"
