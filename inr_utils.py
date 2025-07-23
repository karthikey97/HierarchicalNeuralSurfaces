import torch
import torch.nn as nn
import torch.nn.functional as F
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
