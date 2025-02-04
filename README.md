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
**Mesh Intersection**: Follow the instruction from the official repository https://github.com/vchoutas/torch-mesh-isect \

# Datasets
Download the [Mixamo](https://www.mixamo.com/) dataset and change the path in `data_flags.py`

# Training
Run
```
python main.py
```
to train the skeleton-aware module. Once it is trained, the collision in the resulting animation can be solved by running:
```
python shape_optimization.py
```
Remember to set the correct character name and bvh_path in the `shape_optimization.py` file

