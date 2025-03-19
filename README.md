# MoMa
Official repository for MoMa: Skinned Motion Retargeting Using Masked Pose Modeling. All the code is written using Pytorch Lightning. Please use Pipenv to configure the virtual environment required to run the code.
![image](https://github.com/user-attachments/assets/9161d58b-f83d-4d07-9690-6f5b62d5b97e)

# How to run
Use the following command to configure the virtual environment:
`pipenv install`
# Dependecies
```
pip install -r requirements.txt
```
# Additional Dependecies
**Pytorch 3D**: Follow the instruction from the official repository https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md \
**Blender** : https://docs.blender.org/api/current/info_quickstart.html \
**Mesh Intersection**: Follow the instruction from the official repository https://github.com/vchoutas/torch-mesh-isect 

## 📂 Datasets  

Download the following datasets and organize them in the specified folders:  

### 🏃‍♂️ Mixamo  
- Download animations from **[Mixamo](https://www.mixamo.com/)**.  
- Organize them as:
```plaintext
MIXAMO/
├── Character_1/
│   ├── animation_1.bvh
│   ├── animation_2.bvh
│   └── ...
├── Character_2/
├── Character_3/
└── ...
```
### 🐶 Dog Dataset  
- Download the dog animations from **[AI4Animation](https://github.com/sebastianstarke/AI4Animation)**.  
- Place all `.bvh` files inside:
```plaintext
HumanDog/
├── Dog/
│   ├── animation_1.bvh
│   ├── animation_2.bvh
│   └── ...
```
### 🧍‍♂️ Human Dataset  
- Download human animations from **[Ubisoft LaForge Animation Dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)**.  
- Place all `.bvh` files inside:
```plaintext
HumanDog/
├── Human/
│   ├── animation_1.bvh
│   ├── animation_2.bvh
│   └── ...
```

## 🛠 Setup

Before starting training, configure the dataset settings in [`data_flags.py`](data_flags.py).

### 🔧 Dataset Configuration

Modify the following variables:

- **`dataset_path`**: Path to the dataset directory.
  - **Mixamo Dataset**: The main folder should contain multiple character subfolders, each with `.bvh` animation files.
  - **HumanDog Dataset**: The main folder should contain two subfolders:
    - `Human/` → Contains `.bvh` animation files for human motions.
    - `Dog/` → Contains `.bvh` animation files for dog motions.

- **`dataset`**: Choose the dataset name:
  - `"MIXAMO"` → for the Mixamo dataset.
  - `"HumanDog"` → for the HumanDog dataset.

- **`n_joints`**: Number of joints in the dataset’s superskeleton:
  - `25` → for Mixamo.
  - `26` → for HumanDog.

- **`mode`**: Set the mode to:
  - `"train"` → to enable training.

## 🚀 Training

Once the dataset configuration is set, start training by running:

```
python main.py
```
Once it is trained, the collision in the resulting animation can be solved by running:
```
python shape_optimization.py
```
Remember to set the correct character name and bvh_path in the `shape_optimization.py` file

# Checkpoints
The checkpoints can be downloaded [here](https://drive.google.com/drive/folders/1OE7-mgmSDZizNHU2-CjvWBlq7Hy3P866?usp=sharing)

