# Elixir Replication on Constrained Hardware

This repository documents our replication study of **Elixir: Train a Large Language Model on a Small GPU Cluster** on a much smaller hardware setup than the paper used.

Our goal was **not** to fully reproduce the paper’s multi-A100 throughput numbers. Instead, we focused on bringing up the historical Elixir software stack on a **single GTX 1650-class GPU** and observing how far Elixir’s pipeline could execute before failing.

## What we reproduced

We successfully brought up and exercised the following parts of Elixir:

- distributed initialization
- parameter-order profiling / prefetch order extraction
- chunk planning
- rCache construction
- forward execution for some smaller test cases

## What we could not fully reproduce

We did **not** reproduce the paper’s large-scale performance results. The main reasons were:

- the paper used much larger hardware (A100-based systems)
- our GPU had only about 4 GB VRAM
- the historical Elixir / ColossalAI stack required multiple compatibility fixes
- the original optimizer path (`HybridAdam`) caused extension/toolchain issues on our setup

Because of this, our work should be understood as a **low-resource replication / feasibility study**.

---

# 1. Tested Hardware

This replication was run on:

- NVIDIA GTX 1650-class GPU
- Ubuntu installed on an external SSD
- single-node, single-GPU execution

---

# 2. Repository Dependencies

This reproduction depends on:

- Linux
- Miniforge / Conda
- Python 3.10
- PyTorch 1.13.1 + CUDA 11.6 build
- historical ColossalAI `feature/elixir` branch
- Elixir main repository

---

# 3. Environment Setup

## 3.1 Install system packages

```
sudo apt update
sudo apt install -y git build-essential gcc g++ wget curl
```
## 3.2 Install Miniforge

```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
source ~/.bashrc
```

## 3.3 Create the Conda environment

```
conda create -n elixir python=3.10 -y
conda activate elixir
python -m pip install --upgrade pip setuptools wheel
```
3.4 Install PyTorch 1.13.1 + cu116

```
python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

3.5 Pin NumPy and Transformers

These versions were necessary to avoid compatibility issues with the older Elixir stack.
```
pip install --force-reinstall "numpy==1.26.4"
pip install --force-reinstall "transformers==4.38.2"
```

4. Clone the Required Repositories
  4.1 Clone ColossalAI on the Elixir branch
```
  cd ~
  git clone --branch feature/elixir https://github.com/hpcaitech/ColossalAI.git
```
  4.2 Clone Elixir
  ```
  cd ~
  git clone https://github.com/hpcaitech/Elixir.git
```

5. Required ColossalAI Patch

The historical feature/elixir branch uses a bracketed TCP host format in parallel_context.py that caused launch failures on our system.

  5.1 Apply the patch
  
  From ~/ColossalAI, replace:
  ```
  init_method = f'tcp://[{host}]:{port}'
  ```
  with:
  ```
  init_method = f'tcp://{host}:{port}'
```
  5.2 Example automatic patch
  ```
  cd ~/ColossalAI
  
  python - <<'PY'
  from pathlib import Path
  
  p = Path("colossalai/context/parallel_context.py")
  s = p.read_text()
  
  old = "init_method = f'tcp://[{host}]:{port}'"
  new = "init_method = f'tcp://{host}:{port}'"
  
  if old not in s:
      raise SystemExit("Expected pattern not found.")
  
  p.write_text(s.replace(old, new))
  print("Patched:", p)
  PY
```
  5.3 Install ColossalAI
  ```
  cd ~/ColossalAI
  python -m pip install --no-build-isolation .
```

6. Install Elixir
6.1 Use an older setuptools

This was needed because the historical stack expected pkg_resources.
```
conda activate elixir
python -m pip install "setuptools<82" "wheel" --force-reinstall
```

6.2 Install Elixir
```
cd ~/Elixir
python -m pip install --no-build-isolation .
```

6.3 Verify imports
```
python -c "import torch; print(torch.__version__)"
python -c "import colossalai; print('colossalai ok')"
python -c "import elixir; print('elixir ok')"
```

7. Benchmark Modifications Used in Our Replication

The original Elixir benchmark uses HybridAdam, which triggered cpu_adam extension issues on our setup. To continue the replication, we changed the benchmark to use standard PyTorch Adam.

7.1 Modify example/benchmark/elixir_demo.py

Comment out:
```
from colossalai.nn.optimizer import HybridAdam
```
Replace:
```
optimizer = HybridAdam(model.parameters(), lr=1e-3)
```
with:
```
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```
This change makes the benchmark easier to run on constrained hardware, but it is not identical to the original optimizer path in the paper.

8. Running the Reproductions

Before every run:
```
conda activate elixir
cd ~/Elixir/example/benchmark
export PYTHONPATH=~/Elixir:$PYTHONPATH
mkdir -p elixir_logs
```
All runs were launched manually with torchrun using explicit master_addr and master_port.

8.1 Run 1: gpt2-400m
```
torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  ./elixir_demo.py \
  --model_name=gpt2-400m \
  --batch_size=1 \
  --train_step=2 \
  2>&1 | tee ./elixir_logs/gpt2-400m_gpu_1_bs_1.log
```
Expected result

This run completed initialization, profiling, chunk planning, and rCache setup, then failed with a true CUDA out-of-memory error.

8.2 Run 2: opt-350m
```
torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  ./elixir_demo.py \
  --model_name=opt-350m \
  --batch_size=1 \
  --train_step=2 \
  2>&1 | tee ./elixir_logs/opt-350m_gpu_1_bs_1.log
```
Expected result

This run progressed farther than gpt2-400m, entering forward compute, then failed with CUBLAS_STATUS_NOT_INITIALIZED.

8.3 Run 3: opt-350m with reduced sequence length

Edit example/benchmark/elixir_demo.py and change:
```
SEQ_LEN = 1024
```
to:
```
SEQ_LEN = 256
```
Then rerun:
```
torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  ./elixir_demo.py \
  --model_name=opt-350m \
  --batch_size=1 \
  --train_step=2 \
  2>&1 | tee ./elixir_logs/opt-350m_seq256_gpu_1_bs_1.log
```
Expected result

This still failed during forward-stage compute. Reducing sequence length alone was not enough on our hardware.

8.4 Run 4: custom gpt2-small

We added a smaller GPT-2 configuration because the built-in benchmark models were still too large for a 4 GB GPU.

Step 1: edit example/common/models.py

Add a small-model definition:
```
def gpt2_small():
    return GPTLMModel(hidden_size=768, num_layers=12, num_attention_heads=12)
```
Then add a selector branch:
```
elif name == 'gpt2-small':
    return gpt2_small()
```
Step 2: edit example/benchmark/elixir_demo.py

Set:
```
SEQ_LEN = 128
```
Step 3: run the benchmark
```
torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  ./elixir_demo.py \
  --model_name=gpt2-small \
  --batch_size=1 \
  --train_step=2 \
  2>&1 | tee ./elixir_logs/gpt2-small_gpu_1_bs_1.log
```
Expected result

This was our best run. It successfully completed initialization, profiling, chunk planning, rCache setup, and the first forward pass, then failed during backward propagation with a tensor-shape mismatch.
