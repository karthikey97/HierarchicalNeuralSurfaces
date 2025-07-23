import torch
from pytorch3d.utils import ico_sphere
from pytorch3d.io import save_obj, load_obj
from inr_utils import MLP
import argparse as ap
from mesh_errors import get_face_areas, point2mesh_error
import numpy as np
import torch.nn.functional as F
import time

def parse_args():
    pr = ap.ArgumentParser()
    pr.add_argument("mesh_name", type=str, help="name of the mesh file to be compressed")
    pr.add_argument("--icosphere_level", "-il", type=int, default=7, help="Level of the icosphere")
    pr.add_argument("--hidden_dim", "-hd", type=int, default=44, help="Number of dimensions in the hidden layers")
    pr.add_argument("--num_layers", "-nl", type=int, default=20, help="Number of hidden layers")
    pr.add_argument("--pe_dim", "-pd", type=int, default=10, help="Number of dimensions in the positional encoding")
    return pr.parse_args()

def subdivide(v, f, mask=None):
    if mask is None:
        mask = torch.ones(f.shape[0], device=f.device, dtype=torch.bool)
    unaffected_f = f[~mask]
    affected_f = f[mask]
    faces_as_half_edges = torch.stack([affected_f.roll(-1,1), affected_f],dim=2)
    faces_as_edges = faces_as_half_edges.sort(dim=2)[0]
    faces_as_edges_linear = faces_as_edges.reshape(-1, 2)
    unique_edges, unique_edge_indices = torch.unique(faces_as_edges_linear, dim=0, return_inverse=True)
    midpoints = (v[unique_edges[:,0]] + v[unique_edges[:,1]])/2
    new_v = torch.cat([v, midpoints], dim=0)
    faces_as_edge_idxs = unique_edge_indices.reshape(-1,3) + v.shape[0]
    new_f0 = torch.stack([faces_as_edge_idxs[:,0], affected_f[:,1], faces_as_edge_idxs[:,1]], dim=1)
    new_f1 = torch.stack([faces_as_edge_idxs[:,1], affected_f[:,2], faces_as_edge_idxs[:,2]], dim=1)
    new_f2 = torch.stack([faces_as_edge_idxs[:,2], affected_f[:,0], faces_as_edge_idxs[:,0]], dim=1)
    new_f = torch.cat([unaffected_f, new_f0, new_f1, new_f2, faces_as_edge_idxs], dim=0)
    return new_v, new_f

if __name__=="__main__":
    args = parse_args()
    print(f"Loading coarse model from {f'remeshed/{args.mesh_name}/coarse_weights.pth'}")
    coarse_model = MLP(input_dim=3, hidden_dim=20, output_dim=3, num_layers=12, pe_dim=0).cuda()
    coarse_weights = f'remeshed/{args.mesh_name}/coarse_weights.pth'
    coarse_model.load_state_dict(torch.load(coarse_weights))
    coarse_model.eval()

    print(f"Loading fine model from {f'remeshed/{args.mesh_name}/fine_weights_{args.hidden_dim}_{args.num_layers}.pth'}")
    fine_model = MLP(input_dim=3, hidden_dim=args.hidden_dim, output_dim=3, num_layers=args.num_layers, pe_dim=args.pe_dim).cuda()
    fine_weights = f'remeshed/{args.mesh_name}/fine_weights_{args.hidden_dim}_{args.num_layers}.pth'
    fine_model.load_state_dict(torch.load(fine_weights))
    fine_model.eval()
    
    ics = ico_sphere(args.icosphere_level, device="cuda:0")
    icv = ics.verts_packed()
    icf = ics.faces_packed()
    print(f"Decoding model...")

    start_time = time.time()
    cs = coarse_model(icv)
    while True:
        coarse_face_areas = get_face_areas(cs, icf)
        th = torch.quantile(coarse_face_areas, 0.5) * 4.0
        mask = (coarse_face_areas > th)
        if mask.sum() == 0:
            break
        icv, icf = subdivide(icv, icf, mask)
        cs = coarse_model(icv)
    displacements = fine_model(cs)
    reconstructed = cs + displacements/1414
    total_time = time.time() - start_time
    print(f"Reconstruction complete in {total_time:.2f} seconds")

    del coarse_model, fine_model, icv, cs, displacements
    torch.cuda.empty_cache()

    print("Calculating reconstruction error...")
    ov, of, _ = load_obj(f'remeshed/{args.mesh_name}/input_normalized.obj', load_textures=False, device="cuda:0")
    of = of.verts_idx

    r2t = point2mesh_error(reconstructed, ov, of, scale=1e4)*1e4
    t2r = point2mesh_error(ov, reconstructed, icf, scale=1e4)*1e4
    total_error = r2t + t2r
    print(f'r2t: {r2t:.2f}, t2r: {t2r:.2f}, total_error: {total_error:.2f}')

    print("Saving reconstruction...")
    save_obj(f'remeshed/{args.mesh_name}/reconstruction_{args.hidden_dim}_{args.num_layers}.obj', reconstructed, icf)
