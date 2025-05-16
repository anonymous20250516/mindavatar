# MindAvatar: Full-HD Video & fMRI Dataset of Controllable 3D Avatars for Neural Decoding

The **MindAvatar** dataset pairs high-resolution (1920×1080) videos of controllable 3D digital avatars with whole-brain fMRI recordings. It provides detailed, disentangled annotations for **facial identity, motion, and appearance**, making it well-suited for fine-grained neural decoding tasks at multiple semantic levels.
Each subject in the dataset has over 4.8 hours of video data, covering 2,174 annotated clips, with each clip generated using structured, parametric control over facial geometry and dynamics.

---

## Directory Structure

After downloading and unzipping the dataset, your directory should look like this:

```
${ROOT}
|-- processed
|   |-- sub01
|   |   |-- sub01_fmri_run_norm.npy
|   |-- sub02
|   |   |-- sub02_fmri_run_norm.npy
|   |-- sub03
|   |   |-- sub03_fmri_run_norm.npy
|
|-- stimuli
|   |-- female
|   |   |-- *.mp4         # Videos of female avatars
|   |-- male
|   |   |-- *.mp4         # Videos of male avatars
|   |-- motion_cfgs
|   |   |-- *.npz         # Motion configuration files used to generate the videos
|   |-- id_frames
|       |-- male
|       |   |-- *.jpg     # Canonical portraits for each male avatar identity
|       |-- female
|           |-- *.jpg     # Canonical portraits for each female avatar identity
|
|-- annotation
    |-- sub01_annot.csv    # fMRI-video matching and semantic labels for sub01
    |-- sub02_annot.csv    # Same for sub02
    |-- sub03_annot.csv    # Same for sub03
    |-- id_candidate
        |-- pca_candidate.pt  # Principal components of identity candidates
```

---

## Data Description

* **processed/**: Preprocessed fMRI data for each subject, mapped to the 32k\_fs\_LR cortical surface space.
* **stimuli/male/** and **stimuli/female/**: Full-HD video stimuli of digital avatars.
* **stimuli/id\_frames/**: A canonical portrait (frontal face image) for each avatar identity.
* **stimuli/motion\_cfgs/**: Parameter files that define the motion.
* **annotation/\*.csv**: Files mapping fMRI recordings to the corresponding video clips and providing structured labels.


---
## Installation

Download this repository and create the environment:
```
conda env create -f env.yaml
conda activate mindavatar
pip install hcp_utils
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
```

---
## Evaluation
To evaluate the results on test data, run:
```bash
python eval.py
``` 

---
## Train

We provide scripts to train fMRI-to-avatar mappings for different semantic attributes:

### Structural Identity Reconstruction

Train a model to predict avatar **identity** from fMRI:

```bash
bash train_id.sh
```

### Visual Appearance Modeling

Train models to predict **appearance attributes**:

* **Hair style** from fMRI:

  ```bash
  bash train_hair.sh
  ```

* **Clothing details** from fMRI:

  ```bash
  bash train_cloth_detail.sh
  ```

* **Gender** from fMRI:

  ```bash
  bash train_gender.sh
  ```

You can modify the training scripts or configuration files for customized training.

---

## Pretrained Models

Pretrained checkpoints are available at:

[HuggingFace: Fudan-fMRI / MindAvatar-model](https://huggingface.co/Fudan-fMRI/MindAvatar-model/tree/main)

