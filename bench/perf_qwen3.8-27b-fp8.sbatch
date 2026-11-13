#!/usr/bin/env bash
#SBATCH --job-name=perf-qwen3.8-27b-fp8
#SBATCH --constraint=H200
#SBATCH --gres=gpu:2
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH -o %x-%j.out
set -euo pipefail

pulsarlm_version=0.1.0a9.post1
bench_revision=d9212b07ceaacc8e4b9345666e92f3aab68f208c
uv_version=0.13.7
model=qwen3.8-27b-fp8
checkpoint_revision=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
settings=(full_kv cap_16k cap_16k_mass_0.5)
tensor_parallel_sizes=(2)
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

nvidia-smi
nvidia-smi topo -m

if [[ ! -x $UV_INSTALL_DIR/uv ]]; then
    curl -LsSf "https://astral.sh/uv/$uv_version/install.sh" | sh
fi
uv=$UV_INSTALL_DIR/uv

if [[ ! -x $venv/bin/pulsarlm ]]; then
    "$uv" venv --clear --python 3.13 "$venv"
    "$uv" pip install --python "$venv" "pulsarlm==$pulsarlm_version" httpx==0.28.1 openai==3.31.0 \
        wonderwords==3.0.1 scipy==1.18.1
fi
"$uv" pip freeze --python "$venv"

perf=$work/perf-$bench_revision.py
if [[ ! -f $perf ]]; then
    curl -fsSL "https://raw.githubusercontent.com/cosmonomic/pulsarlm/$bench_revision/bench/perf.py" -o "$perf.part"
    mv "$perf.part" "$perf"
fi
check=$work/tp_check-$bench_revision.py
if [[ ! -f $check ]]; then
    curl -fsSL "https://raw.githubusercontent.com/cosmonomic/pulsarlm/$bench_revision/bench/tp_check.py" -o "$check.part"
    mv "$check.part" "$check"
fi

if [[ ! -d $PULSARLM_DATA_DIR/$model ]]; then
    "$venv/bin/pulsarlm" models download "$model" --revision "$checkpoint_revision"
else
    "$venv/bin/pulsarlm" models migrate "$model" "$PULSARLM_DATA_DIR/$model"
fi
rm -rf "$HF_HOME"

if [[ ! -f $work/tp_check/tp_check.json ]]; then
    rm -rf "$work/tp_check"
    "$venv/bin/python" -I "$check" \
        --model "$model" \
        --out "$work/tp_check" \
        --sizes 1 \
        --sizes 2 \
        --max-tokens 64 \
        --kv-cache-gib "$kv_cache_gib"
fi

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
