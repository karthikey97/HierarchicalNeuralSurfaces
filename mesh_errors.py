import torch
import torch.nn.functional as F
from pytorch3d._C import point_face_dist_forward

def get_face_areas(v,f):
    f_norms = torch.cross(v[f[:,0]]-v[f[:,1]], v[f[:,0]]-v[f[:,2]], dim=1)
    f_areas = torch.sum(f_norms**2, dim=1) **0.5 * 0.5
    return f_areas


def sample_points_on_mesh(v,f,num_pts, prior = None):
    if prior is None:
        f_areas = get_face_areas(v,f)
        probs = F.normalize(f_areas, dim=0, p=1)
    else:
        probs = F.normalize(prior, dim=0, p=1)
    num_pts = int(num_pts)
    sampled_f_idxs = torch.multinomial(probs, num_pts, replacement=True)
    sampled_f = f[sampled_f_idxs]
    alpha = torch.rand(num_pts,1).to(v.device)
    beta = torch.rand(num_pts,1).to(v.device)
    k = beta**0.5
    a = 1 - k
    b = (1-alpha) * k
    c = alpha * k
    sampled_pts = v[sampled_f[:,0],:]*a + v[sampled_f[:,1],:]*b + v[sampled_f[:,2],:]*c
    return sampled_pts, sampled_f, (a,b,c)


def point2mesh_error(dv, ov, of, scale = 1.0):
    orig_device = ov.device.type
    if 'cuda' not in orig_device:
        ov = ov.to("cuda:0")
        of = of.to("cuda:0")
    dpcl = dv * scale
    ov = ov*scale
    min_area = torch.min(get_face_areas(ov,of)).item()
    i1 = torch.LongTensor([0]).to(dpcl.device)
    dists, idxs = point_face_dist_forward(dpcl, i1, ov[of], i1, dpcl.size(0), min_area)
    errors = torch.sqrt(dists)
    return errors.mean()/scale
