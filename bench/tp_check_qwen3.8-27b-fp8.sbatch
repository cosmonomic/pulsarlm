#!/usr/bin/env bash
#SBATCH --job-name=tp-check-qwen3.8-27b-fp8
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-06:00:00
#SBATCH -o %x-%j.out
set -euo pipefail

pulsarlm_version=0.1.0a9.post1
pulsarlm_tag=v0.1.0-alpha9-post1
uv_version=0.13.7
model=qwen3.8-27b-fp8
checkpoint_revision=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a

work=${WORK_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/tp-check-$model}
venv=$work/venv
mkdir -p "$work"

export UV_INSTALL_DIR=$work/uv
export UV_NO_MODIFY_PATH=1
export UV_CACHE_DIR=$work/uv-cache
export UV_PYTHON_INSTALL_DIR=$work/python
export HF_HOME=$work/hf
export XDG_CACHE_HOME=$work/cache
export PULSARLM_DATA_DIR=$work/data

nvidia-smi
nvidia-smi topo -m

if [[ ! -x $UV_INSTALL_DIR/uv ]]; then
    curl -LsSf "https://astral.sh/uv/$uv_version/install.sh" | sh
fi
uv=$UV_INSTALL_DIR/uv

if [[ ! -x $venv/bin/pulsarlm ]]; then
    "$uv" venv --python 3.13 "$venv"
    "$uv" pip install --python "$venv" "pulsarlm==$pulsarlm_version" httpx==0.28.1 openai==3.31.0 wonderwords==3.0.1
fi
"$uv" pip freeze --python "$venv"

check=$work/tp_check-$pulsarlm_tag.py
if [[ ! -f $check ]]; then
    curl -fsSL "https://raw.githubusercontent.com/cosmonomic/pulsarlm/$pulsarlm_tag/bench/tp_check.py" -o "$check"
fi

if [[ ! -d $PULSARLM_DATA_DIR/$model ]]; then
    "$venv/bin/pulsarlm" models download "$model" --revision "$checkpoint_revision"
else
    "$venv/bin/pulsarlm" models migrate "$model" "$PULSARLM_DATA_DIR/$model"
fi
rm -rf "$HF_HOME"

"$venv/bin/python" -I "$check" \
    --model "$model" \
    --out "$work/results" \
    --sizes 1 \
    --sizes 2 \
    --kv-cache-gib 8
