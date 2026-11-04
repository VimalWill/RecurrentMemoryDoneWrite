#!/bin/bash
# Usage: ./train_mqar.sh            run the sweep one model at a time
#        ./train_mqar.sh -p         run it in parallel across GPUs with ray
# Options such as MQAR_MIXERS are listed in mqar_configs.py.
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD/RecurrentMemory/src${PYTHONPATH:+:$PYTHONPATH}"

exec "${PYTHON:-python}" -m zoology.launch mqar_configs.py --name "${NAME:-mqar-recall}" "$@"
