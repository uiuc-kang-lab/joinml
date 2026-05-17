#!/bin/bash
# Acquire the datasets used in the JoinML preprint.
#
# Usage:
#   bash reproduction/setup/download_data.sh           # interactive: lists and downloads all paper datasets
#   bash reproduction/setup/download_data.sh quora     # one dataset by name
#   bash reproduction/setup/download_data.sh check     # audit which datasets are present, no downloads
#
# Paper datasets (from results.py):
#
#   Auto-downloadable via Google Drive (gdown):
#     quora, company, flickr30k
#
#   Manual acquisition (license / external host):
#     VeRi, roxford, webmasters
#
#   SemBench-derived (regenerated via the SemBench upstream):
#     ecomm-q7, ecomm-q8, ecomm-q9, ecomm-q10, ecomm-q11, movie-q5, movie-q6

set -euo pipefail
cd "$(dirname "$0")/../.."

DATA_DIR=data
PAPER_AUTODL=(quora company flickr30k)
PAPER_MANUAL=(VeRi roxford webmasters)
PAPER_SEMBENCH=(ecomm-q7 ecomm-q8 ecomm-q9 ecomm-q10 ecomm-q11 movie-q5 movie-q6)
ALL_PAPER=("${PAPER_AUTODL[@]}" "${PAPER_MANUAL[@]}" "${PAPER_SEMBENCH[@]}")

# --- per-dataset acquisition handlers -------------------------------------- #

manual_instructions() {
  case "$1" in
    VeRi)
      cat <<'EOF'
        Source : https://github.com/JDAI-CV/VeRidataset (request access via the README's contact)
        Layout : data/VeRi/data/table0.csv + data/VeRi/oracle_labels/00.csv
                 (text-only join_col extracted from the vehicle metadata
                  CSV; oracle labels from the multi-camera tracking GT)
EOF
      ;;
    roxford)
      cat <<'EOF'
        Source : http://cmp.felk.cvut.cz/revisitop/data/datasets/roxford5k/
        Layout : data/roxford/data/table0.csv + data/roxford/oracle_labels/00.csv
                 (rOxford5k images; embeddings via the SuperGlobal proxy)
EOF
      ;;
    webmasters)
      cat <<'EOF'
        Source : Stack Exchange Data Dump → Webmasters site (https://archive.org/details/stackexchange)
                 Pre-processing per Zhang et al. 2015 (Multi-factor Duplicate Question
                 Detection in Stack Overflow).
        Layout : data/webmasters/data/table0.csv + data/webmasters/oracle_labels/00.csv
EOF
      ;;
  esac
}

download_auto() {
  local ds=$1
  if [ -d "$DATA_DIR/$ds" ]; then
    echo "  ✓ data/$ds already present"
    return 0
  fi
  echo "  → downloading data/$ds via data/download.sh (gdown)"
  (cd "$DATA_DIR" && bash download.sh download "$ds") \
    || { echo "  ✗ download failed; see data/download.sh"; return 1; }
}

handle_manual() {
  local ds=$1
  if [ -d "$DATA_DIR/$ds" ]; then
    echo "  ✓ data/$ds already present"
    return 0
  fi
  echo "  ✗ data/$ds NOT present — manual acquisition required:"
  manual_instructions "$ds"
}

handle_sembench() {
  local ds=$1
  if [ -d "$DATA_DIR/$ds" ]; then
    echo "  ✓ data/$ds already present"
    return 0
  fi
  echo "  ✗ data/$ds NOT present — see data/SEMBENCH.md to regenerate from the"
  echo "    SemBench upstream (https://github.com/SemBench/SemBench/)."
}

handle() {
  local ds=$1
  case " ${PAPER_AUTODL[*]} " in *" $ds "*) download_auto "$ds"; return $?;; esac
  case " ${PAPER_MANUAL[*]} " in *" $ds "*) handle_manual "$ds"; return 0;; esac
  case " ${PAPER_SEMBENCH[*]} " in *" $ds "*) handle_sembench "$ds"; return 0;; esac
  echo "  ✗ '$ds' is not a recognised paper dataset"
  echo "    Known: ${ALL_PAPER[*]}"
  return 2
}

# --- CLI dispatch ---------------------------------------------------------- #

if [ $# -eq 0 ] || [ "${1:-}" = "all" ] || [ "${1:-}" = "paper" ]; then
  echo "[setup] Acquiring paper datasets into $DATA_DIR/ ..."
  failed=()
  for ds in "${ALL_PAPER[@]}"; do
    echo "[$ds]"
    handle "$ds" || failed+=("$ds")
  done
  echo
  if [ ${#failed[@]} -ne 0 ]; then
    echo "Warning: ${#failed[@]} dataset(s) need manual acquisition: ${failed[*]}"
  fi
  echo "[setup] Done. To audit: bash reproduction/setup/download_data.sh check"
elif [ "${1:-}" = "check" ]; then
  echo "[setup] Paper-dataset presence audit:"
  missing=()
  for ds in "${ALL_PAPER[@]}"; do
    if [ -d "$DATA_DIR/$ds" ]; then
      echo "  ✓ $ds"
    else
      echo "  ✗ $ds"
      missing+=("$ds")
    fi
  done
  echo
  if [ ${#missing[@]} -eq 0 ]; then
    echo "All ${#ALL_PAPER[@]} paper datasets present."
  else
    echo "Missing (${#missing[@]}): ${missing[*]}"
    echo "Run 'bash reproduction/setup/download_data.sh' for acquisition instructions."
    exit 1
  fi
else
  for arg in "$@"; do
    echo "[$arg]"
    handle "$arg" || true
  done
fi
