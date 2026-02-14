# mlp_qmmm

Long-range **MLP QM/MM framework** for machine learning potentials with explicit handling of QM atoms and embedded MM charges.  
Implements descriptor + electrostatic feature networks, flexible training via YAML configs, and reproducible evaluation workflows.

---

## 📦 Installation

Clone the repository and install into your environment:

```bash
#Conda Environment
conda create -n  mlpqmmmenv python==3.13   #should be python3 <=3.13
conda activate mlpqmmmenv
#conda install conda-forge::mamba
#mamba install -c conda-forge ipython numpy scipy matplotlib mdanalysis multiprocess tqdm pandas pyyaml
pip install ipython numpy scipy matplotlib multiprocess tqdm pandas pyyaml setuptools

# pip install torch # based on the cuda in the machine you are working with
# For CUDA 12
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

#For CUDA 11
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# or try # pip install torch torchvision torchaudio

cd <package_base_dir>
pip install -e .
```
---
## 🚀 Quick Start
### 1. How to Train
Prepare a YAML config file (e.g., `config.yaml`) with the desired settings.
Then run the training script:
```bash
qmmm-train config.yaml
```
### 2. How to Asess Performance
After training, evaluate the model using:
```bash
qmmm-test <saved_dir>
```

## ✨ Authors

- **Abdul Raafik Arattu Thodika**  
- **Xiaoliang Pan**

---

## 📜 License

See [LICENSE](LICENSE).
