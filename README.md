<h1 align="center">🪩 CustomDance</h1>
<h3 align="center">Customized 3D Dance Generation with Coarse-to-Fine Human-Centered Interactive Control</h3>

<p align="center">
  <a href="https://arxiv.org/pdf/2608.06722">
    <img src="https://img.shields.io/badge/Paper-CustomDance-b31b1b" alt="Paper">
  </a>
  <a href="https://xulongt.github.io/customdance-project-page/">
    <img src="https://img.shields.io/badge/Project_Page-CustomDance-blue" alt="Project Page">
  </a>
  <a href="https://asia.siggraph.org/2026/">
    <img src="https://img.shields.io/badge/Conference-SIGGRAPH%20Asia%202026-orange" alt="Conference">
  </a>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/e889eb42-9515-4fc5-bfd0-dcb85ba5165b" width="100%" controls playsinline preload="metadata" aria-label="CustomDance demo video"></video>
</p>

> **Abstract**: With the rise of AI-generated content (AIGC) and advanced techniques for 3D human representation, the task of generating 3D dance movements has become an exciting area of research. Despite significant advancements, current methods often fail to provide comprehensive and distinct control over various multimodal inputs from users, such as music or specific descriptions of desired movements. As a result, the generated motions may be statistically plausible and technically correct, but they often lack depth, expressiveness, and alignment with the user's creative vision. To address this issue, we present CustomDance, a coarse-to-fine interactive system designed for customized 3D dance generation. Inspired by the workflows of expert choreographers, CustomDance introduces a novel paradigm to AI-assisted choreography through three interconnected stages. First, a multimodal Large Language Model (MLLM) analyzes the music and a high-level text prompt to identify key temporal anchors and creative cues for the piece. Next, for each anchor, a multimodal retriever suggests high-quality motion clips from a dance library based on local music and text, empowering the user with concrete and predictable options. Finally, a custom music-conditioned diffusion in-painter seamlessly connects the selected phrases, allowing for iterative, user-guided refinement of the final composition, supported by visualizations of motion dynamics.

<p align="center"><strong>🎉 CustomDance has been accepted to SIGGRAPH Asia 2026! 🎉</strong></p>

## 🚀 Code

### 🛠️ Set up the Environment

**1. Create the environment.**

```bash
conda create -n customdance python=3.10.18 pip -y
conda activate customdance
sudo apt update
sudo apt install -y build-essential ffmpeg libsndfile1 curl
```

**2. Install PyTorch (CUDA 12.8) and build dependencies.**

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c requirements.txt numpy==1.23.5 setuptools wheel packaging ninja psutil
```

**3. Install CUDA extensions and CustomDance.**

```bash
MAX_JOBS=4 NVCC_THREADS=2 CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE \
  python -m pip install --no-build-isolation -c requirements.txt \
  causal-conv1d==1.5.3.post1 mamba-ssm==2.2.6.post3
python -m pip install --no-build-isolation -r requirements.txt
python -m pip install -e . --no-deps
```

### 📦 Download Resources

| Resource | Location |
| --- | --- |
| [FineDance dataset](https://github.com/li-ronghui/FineDance#download-the-finedance-dataset) | `data/finedance/` |
| [Joint Reference](https://raw.githubusercontent.com/li-ronghui/FineDance/0476cd42619e57fb6ec4840e81acaf7743b1507d/smplx_neu_J_1.npy) | `data/reference/smplx_neu_J_1.npy` |
| [SMPL Model](https://smpl.is.tue.mpg.de/) | `assets/smpl/SMPL_NEUTRAL.pkl` |
| [CustomDance Resources](https://drive.google.com/file/d/1qgXu4XvDQ8T5aA9_gyoU-QFkzlwx_X3s/view?usp=sharing) | `assets/runtime/` |

For SMPL, rename `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`.
The CustomDance bundle includes `stage3_model.pt` (inpainting), `stage2_model.pt` (retriever),
the condition normalizer and the paired retrieval index.

After downloading the bundle, extract it into `assets/runtime/`:

```bash
mkdir -p assets/runtime
tar -xzf /path/to/customdance-resources-v0.1.tar.gz -C assets/runtime
```

### 🧩 Prepare the Data

```bash
python scripts/prepare_data.py \
  --raw-root data/finedance \
  --rest-joints data/reference/smplx_neu_J_1.npy
```

### 📂 Directory Structure

After downloading the resources and preparing the data:

```text
CustomDance/
├── app/
│   ├── utils/                 # Retriever, inpainting, Diagnoser and helpers
│   ├── models/                # Retriever and inpainting networks
│   ├── preprocessing/         # FineDance filtering and library construction
│   ├── api/                   # Backend routes
│   ├── frontend/              # SMPL editor
│   ├── schemas/               # Motion and API contracts
│   ├── config.py
│   ├── runtime.py
│   ├── sessions.py
│   └── timeline.py
├── configs/
├── scripts/
│   ├── prepare_data.py
│   ├── check_install.py
│   └── serve.py
├── data/
│   ├── finedance/
│   │   ├── label_json/
│   │   ├── motion/
│   │   ├── music_npy/
│   │   └── music_wav/
│   └── reference/smplx_neu_J_1.npy
├── assets/
│   ├── smpl/SMPL_NEUTRAL.pkl
│   └── runtime/
│       ├── inpainting/
│       │   ├── stage3_model.pt
│       │   └── condition_normalizer.npz
│       ├── retriever/
│       │   ├── stage2_model.pt
│       │   └── index/
│       │       ├── embeddings.npy
│       │       ├── metadata.jsonl
│       │       ├── index.json
│       │       └── summary.json
│       └── motion_library/    # Generated by prepare_data.py
├── outputs/                   # Created when using the app
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── ASSET_MANIFEST.json
├── LICENSE
└── README.md
```

### ▶️ Run CustomDance

Create your local configuration:

```bash
cp -n .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then start the app:

```bash
python scripts/serve.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
Intent analysis requires OpenAI API access, billed separately. Keep `.env` private.

### 🎵 Usage

1. Import a **WAV file**, enter **Global Intent**, allow API processing and click **Analyze Music**.
2. Select a timeline slot, edit **Local Intent** and click **Refresh Motions**. Choose a candidate and click **Use**.
3. Repeat for the other slots, then click **Complete** to connect the selected motions and preview the dance with SMPL.
4. To refine the result, enter **Repair**, run **Diagnose**, then select a time range and joint groups. **Smooth** interpolates only the selected group; selecting multiple groups enables **Remake**, which regenerates the whole body within the selected range.
5. Click **PKL** to download the motion. A copy is also saved under `outputs/sessions/<session_id>/exports/`.

## 🙏 Acknowledgements

This code is standing on the shoulders of giants. We thank the contributors of
[FlowerDance](https://github.com/XulongT/FlowerDance),
[TMR](https://github.com/Mathux/TMR),
[Mamba](https://github.com/state-spaces/mamba), and
[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d).
Our data preprocessing builds on [FineDance](https://github.com/li-ronghui/FineDance),
and our human model is provided by [SMPL](https://smpl.is.tue.mpg.de/).

## 📄 Citation

```bibtex
@article{tang2026customdance,
  title={CustomDance: Customized 3D Dance Generation with
         Coarse-to-Fine Human-Centered Interactive Control},
  author={Tang, Xulong and Yang, Kaixing and Guo, Xiaohu and
          Prabhakaran, Balakrishnan and Alghofaili, Rawan},
  journal={arXiv preprint arXiv:2608.06722},
  year={2026}
}
```

---

[License](LICENSE)
