## Instructions:
0. Download the `tar.gz` original meshes and their compressed binaries [here](https://drive.google.com/file/d/1sBGOD-hfeKnAb6ivziJb8KeeBNUboMIS/view?usp=sharing). 
1. Install `pytorch3d` and all its dependencies.
2. Run `python {mesh_name} --hidden_dims {hd} --num_layers {nl}`
   For Example: `python armadillo --hidden_dims 35 --num_layers 16`

Important note: Please ensure that `{mesh_name}` directory exists in `remeshed/`, and it should contain `coarse_weights.bin` and `fine_weights_{hd}_{nl}.bin`.
