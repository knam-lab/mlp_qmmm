# mlp_qmmm 0.x

Long-range **MLP QM/MM framework** for machine learning potentials with explicit handling of QM atoms and embedded MM charges.  
Implements descriptor + electrostatic feature networks, flexible training via YAML configs, and reproducible evaluation workflows.

### References

[1] Pan, X.; Yang, J.; Van, R.; Epifanovsky, E.; Ho, J.; Huang, J.; Pu, J.; Mei, Y.; Nam, K.; Shao, Y. Machine-Learning-Assisted Free Energy Simulation of Solution-Phase and Enzyme Reactions. J. Chem. Theory Comput. 2021, 17 (9), 5745–5758.

[2] Arattu Thodika, A. R.; Pan, X.; Shao, Y.; Nam, K. Machine Learning Quantum Mechanical/Molecular Mechanical Potentials: Evaluating Transferability in Dihydrofolate Reductase-Catalyzed Reactions. J. Chem. Theory Comput. 2025, 21 (2), 817–832.

[3] Arattu Thodika, A. R.; Panda, S. K.; Pan, X.; Shao, Y.; Nam, K. Accurate and Time-Efficient Condensed-Phase Free Energy Simulations with Reaction-Specific &Delta;-Machine Learning Potentials in CHARMM, *under review*

---

## 📦 Installation

Clone/Download the repository and install into your environment:

```bash
#Conda Environment
conda create -n  mlpqmmmenv python==3.13   #should be python3 <=3.13 (for cuda 12-6 torch)
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

cd <package_base_dir>    # has .toml file and src
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
