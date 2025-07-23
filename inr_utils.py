import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pytorch3d.io import load_obj
from mesh_errors import sample_points_on_mesh, get_face_areas
from copy import deepcopy
import numpy as np

class PE(nn.Module):
    def __init__(self, pe_dim):
        super(PE, self).__init__()
        self.pe_dim = pe_dim
    def forward(self, x):
        if self.pe_dim == 0:
            return x
        st = [x]
        for j in range(self.pe_dim):
            st.append(torch.sin((2**j)*x*torch.pi))
            st.append(torch.cos((2**j)*x*torch.pi))
        return torch.cat(st, dim=1)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, pe_dim):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.pe = PE(pe_dim)
        self.layers = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim*(pe_dim*2+1) if i == 0 else hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
        ) for i in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.pe(x) if self.pe is not None else x
        out = x
        for i in range(self.num_layers):
            residual = out
            out = self.layers[i](out)
            if i>0: out += residual
        out = self.output_layer(out)
        return out
    def serialize_params(self, file_path):  
        param_names = []
        param_values = []
        total_elements = 0
        for name, param in self.named_parameters():
            param_names.append(name)
            param_values.append(param.data.cpu())
            total_elements += param.data.nbytes
        with open(file_path, 'wb') as f:
            for value in param_values:
                bytes = value.numpy().tobytes()
                f.write(bytes)
        return total_elements
    def read_from_bin(self, file_path, read_type="float16"):
        with open(file_path, 'rb') as f:    
            np_bytes = np.frombuffer(f.read(), dtype=read_type)
        offset = 0
        for name, param in self.named_parameters():
            try:
                num_elements_to_read = param.data.numel()
                param.data = torch.tensor(np_bytes[offset:offset+num_elements_to_read]).reshape(param.data.shape).to(param.data.device).to(param.data.dtype)
                offset += num_elements_to_read
            except:
                print(f"Error reading {name}")
                breakpoint()


class CoarseDataset(Dataset):
    def __init__(self, mesh_name, batch_size=2048):
        embedding_path = f'remeshed/{mesh_name}/embedding.obj'
        normalized_path = f'remeshed/{mesh_name}/input_normalized.obj'
        self.embedding, embedding_faces, _ = load_obj(embedding_path, load_textures=False, device="cpu")
        self.normalized, normalized_faces, _ = load_obj(normalized_path, load_textures=False, device="cpu")
        self.embedding_faces = embedding_faces.verts_idx
        self.normalized_faces = normalized_faces.verts_idx
        self.batch_size = batch_size
    def __len__(self):
        return 100
    def __getitem__(self, idx):
        sampled_pts, sampled_f, bcs = sample_points_on_mesh(self.embedding, self.embedding_faces, self.batch_size)
        svf = self.normalized[sampled_f]
        sampled_targets = svf[:,0,:]*bcs[0] + svf[:,1,:]*bcs[1] + svf[:,2,:]*bcs[2]
        return sampled_pts, sampled_targets

class FineDataset(Dataset):
    def __init__(self, coarse_dataset, coarse_model, batch_size=2024, scale=1.0):
        self.coarse_dataset = coarse_dataset
        with torch.no_grad():
            try:
                coarse_reconstruction = coarse_model(self.coarse_dataset.embedding.cuda())
            except:
                emb_chunks = torch.chunk(self.coarse_dataset.embedding, 100, dim=0)
                coarse_reconstruction = torch.cat([coarse_model(emb.cuda()) for emb in emb_chunks], dim=0)
        self.prior = get_face_areas(coarse_reconstruction, self.coarse_dataset.embedding_faces.cuda())
        self.prior = F.normalize(self.prior, dim=0, p=1).cpu()
        self.coarse_model = deepcopy(coarse_model).cpu()
        self.scale = scale
        self.batch_size = batch_size
    def __len__(self):
        return 100
    def __getitem__(self, idx):
        sampled_pts, sampled_f, bcs = sample_points_on_mesh(self.coarse_dataset.embedding, 
                            self.coarse_dataset.embedding_faces, 
                            self.batch_size, self.prior)
        with torch.no_grad():
            sampled_coarse = self.coarse_model(sampled_pts)
        sfv = self.coarse_dataset.normalized[sampled_f]
        sampled_targets = sfv[:,0,:]*bcs[0] + sfv[:,1,:]*bcs[1] + sfv[:,2,:]*bcs[2]
        return sampled_coarse, (sampled_targets-sampled_coarse)*self.scale

