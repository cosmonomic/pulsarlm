#!/usr/bin/env bash
#SBATCH --job-name=perf-qwen3.8-27b-fp8
#SBATCH --nodelist=watgpu308
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH -o %x-%j.out
set -euo pipefail

pulsarlm_version=0.1.0a5
pulsarlm_tag=v0.1.0-alpha-5
uv_version=0.13.7
model=qwen3.8-27b-fp8
checkpoint_revision=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
settings=(full_kv cap_16k)
tensor_parallel_sizes=(1 2)
concurrencies=(1 4)
runs=5
kv_cache_gib=18
max_context=131072

work=${WORK_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/perf-$model}
venv=$work/venv
mkdir -p "$work"

export UV_INSTALL_DIR=$work/uv
export UV_NO_MODIFY_PATH=1
export UV_CACHE_DIR=$work/uv-cache
export UV_PYTHON_INSTALL_DIR=$work/python
export HF_HOME=$work/hf
export XDG_CACHE_HOME=$work/cache
export PULSARLM_DATA_DIR=$work/data

gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ $gpus != $'NVIDIA RTX A6000\nNVIDIA RTX A6000' ]]; then
    echo "expected two NVIDIA RTX A6000, got: $gpus" >&2
    exit 1
fi
nvidia-smi
nvidia-smi topo -m

if [[ ! -x $UV_INSTALL_DIR/uv ]]; then
    curl -LsSf "https://astral.sh/uv/$uv_version/install.sh" | sh
fi
uv=$UV_INSTALL_DIR/uv

if [[ ! -x $venv/bin/pulsarlm ]]; then
    "$uv" venv --python 3.13 "$venv"
    "$uv" pip install --python "$venv" "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu132
    "$uv" pip install --python "$venv" "pulsarlm==$pulsarlm_version" httpx==0.28.1 openai==3.31.0 \
        wonderwords==3.0.1 scipy==1.18.1
fi
"$uv" pip freeze --python "$venv"

perf=$work/perf-$pulsarlm_tag.py
if [[ ! -f $perf ]]; then
    curl -fsSL "https://raw.githubusercontent.com/cosmonomic/pulsarlm/$pulsarlm_tag/bench/perf.py" -o "$perf"
fi

if [[ ! -f $PULSARLM_DATA_DIR/$model.pulsarlm ]]; then
    "$venv/bin/pulsarlm" download "$model" --revision "$checkpoint_revision"
fi
rm -rf "$HF_HOME"

for setting in "${settings[@]}"; do
    for tensor_parallel_size in "${tensor_parallel_sizes[@]}"; do
        for concurrency in "${concurrencies[@]}"; do
            "$venv/bin/python" -I "$perf" run \
                --setting "$setting" \
                --model "$model" \
                --out "$work/results" \
                --concurrency "$concurrency" \
                --tensor-parallel-size "$tensor_parallel_size" \
                --runs "$runs" \
                --max-context "$max_context" \
                --kv-cache-gib "$kv_cache_gib"
        done
    done
done

"$venv/bin/python" -I "$perf" plot --results "$work/results/results.jsonl" --out "$work/results/perf.svg"
