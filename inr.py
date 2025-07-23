import torch
import torch.nn as nn
from tqdm import tqdm
from inr_utils import MLP, CoarseDataset, FineDataset
import argparse as ap
from pytorch3d.io import save_obj
from pytorch3d.utils import ico_sphere
import torch.optim as optim
from torch.utils.data import DataLoader
from mesh_errors import point2mesh_error
from reconstruct import reconstruct
import os

def parse_args():
    pr = ap.ArgumentParser()
    pr.add_argument("mesh_name", type=str, help="name of the mesh file to be compressed")
    pr.add_argument("--icosphere_level", "-il", type=int, default=8, help="Level of the icosphere")
    pr.add_argument("--hidden_dim", "-hd", type=int, default=56, help="Number of dimensions in the hidden layers")
    pr.add_argument("--num_layers", "-nl", type=int, default=20, help="Number of hidden layers")
    pr.add_argument("--scale", "-os", type=float, default=1414, help="Scale factor of the outputs")
    pr.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate")
    pr.add_argument("--fine_epochs", "-ne", type=int, default=2000, help="Number of epochs")
    pr.add_argument("--batch_size", "-bs", type=int, default=2048, help="Batch size")
    pr.add_argument("--coarse_epochs", "-ce", type=int, default=2000, help="Number of epochs for the coarse model")
    return pr.parse_args()

if __name__=="__main__":

    args = parse_args() 

    if not os.path.exists(f'remeshed/{args.mesh_name}'):
        os.system(f'../smat/build/coarse_to_fine {args.mesh_name}')
        assert os.path.exists(f'remeshed/{args.mesh_name}'), f'remeshed/{args.mesh_name} does not exist'
        assert os.path.exists(f'remeshed/{args.mesh_name}/embedding.obj'), f'remeshed/{args.mesh_name}/embedding.obj does not exist'
    
    coarse_dataset = CoarseDataset(args.mesh_name, args.batch_size)
    loader = DataLoader(coarse_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    coarse_model = MLP(input_dim=3, hidden_dim=20, output_dim=3, num_layers=12, pe_dim=0).cuda()
    criterion = nn.MSELoss()

    if os.path.exists(f'remeshed/{args.mesh_name}/coarse_weights.pth'):
        coarse_model.load_state_dict(torch.load(f'remeshed/{args.mesh_name}/coarse_weights.pth'))
        print(f'Loaded coarse model from "remeshed/{args.mesh_name}/coarse_weights.pth"')
    else:
        print(f'Training coarse model for {args.coarse_epochs} epochs')
        optimizer = optim.AdamW(coarse_model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.coarse_epochs)
        pbar = tqdm(total=args.coarse_epochs, ncols=150)
        for epoch in range(args.coarse_epochs):
            running_loss = 0.0
            for inputs, targets in loader:
                inputs = inputs[0].cuda()
                targets = targets[0].cuda()
                optimizer.zero_grad()
                outputs = coarse_model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()*inputs.size(0)
                loss.backward()
                optimizer.step()
            pbar.update(1)
            pbar.set_description(f'Epoch {epoch+1}/{args.coarse_epochs} Loss: {running_loss/len(loader):.4f}')
        pbar.close()
        coarse_model = coarse_model.half().float()
        torch.save(coarse_model.state_dict(), f'remeshed/{args.mesh_name}/coarse_weights.pth')
        with torch.no_grad():
            ics = ico_sphere(args.icosphere_level, device="cuda:0")
            icv = ics.verts_packed()
            icf = ics.faces_packed()
            cs = coarse_model(icv)
            save_obj(f'remeshed/{args.mesh_name}/coarse_reconstruction.obj', cs, icf)
            print(f'Saved coarse reconstruction to "remeshed/{args.mesh_name}/coarse_reconstruction.obj"')
        del ics, icv, icf, cs, optimizer, scheduler, loader
    
    fine_dataset = FineDataset(coarse_dataset, coarse_model, args.batch_size, args.scale)
    loader = DataLoader(fine_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    fine_model = MLP(input_dim=3, hidden_dim=args.hidden_dim, output_dim=3, num_layers=args.num_layers, pe_dim=10).cuda()
    optimizer = optim.AdamW(fine_model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.fine_epochs)
    ics = ico_sphere(args.icosphere_level, device="cuda:0")
    icv = ics.verts_packed()
    icf = ics.faces_packed()
    del ics
    torch.cuda.empty_cache()
    cs = coarse_model(icv)
    lowest_error = 500000000

    total_error = 10.00
    pbar = tqdm(total=args.fine_epochs, ncols=150)
    for epoch in range(args.fine_epochs):
        fine_model.train()
        running_loss = 0.0
        for inputs, targets in loader:
            inputs = inputs[0].cuda()
            targets = targets[0].cuda()
            optimizer.zero_grad()
            outputs = fine_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(loader)
        scheduler.step()
        if (epoch+1)%20==0:
            fine_model = fine_model.half().float()
        
        if ((epoch+1)%100)==0:
            fine_model.eval()
            with torch.no_grad():
                displacements = fine_model(cs)
                reconstructed = cs + displacements/args.scale
            r2t = point2mesh_error(reconstructed, coarse_dataset.normalized.cuda(), coarse_dataset.normalized_faces.cuda(), scale=1e4)*1e4
            t2r = point2mesh_error(coarse_dataset.normalized.cuda(), reconstructed, icf, scale=1e4)*1e4
            total_error = r2t + t2r
            if total_error < lowest_error:
                lowest_error = total_error
                torch.save(fine_model.state_dict(), f'remeshed/{args.mesh_name}/fine_weights_{args.hidden_dim}_{args.num_layers}.pth')
            torch.cuda.empty_cache()
        pbar.update(1)
        pbar.set_description(f'Epoch {epoch+1}/{args.fine_epochs} Loss: {epoch_loss:.4f} Total error: {total_error:.2f}')
    pbar.close()
    fine_model.load_state_dict(torch.load(f'remeshed/{args.mesh_name}/fine_weights_{args.hidden_dim}_{args.num_layers}.pth'))

    reconstructed, icf = reconstruct(coarse_model, fine_model, args.icosphere_level)


    r2t = point2mesh_error(reconstructed, coarse_dataset.normalized.cuda(), coarse_dataset.normalized_faces.cuda(), scale=1e4)*1e4
    t2r = point2mesh_error(coarse_dataset.normalized.cuda(), reconstructed, icf, scale=1e4)*1e4
    total_error = r2t + t2r
    print(f"Total error: {total_error:.2f}, r2t: {r2t:.2f}, t2r: {t2r:.2f}")
    save_obj(f'remeshed/{args.mesh_name}/reconstruction_{args.hidden_dim}_{args.num_layers}.obj', reconstructed, icf)
    print(f'Saved reconstruction to "remeshed/{args.mesh_name}/reconstruction_{args.hidden_dim}_{args.num_layers}.obj"')

    coarse_model = coarse_model.half()
    coarse_size = coarse_model.serialize_params(f'remeshed/{args.mesh_name}/coarse_weights.bin')/1024
    fine_model = fine_model.half()
    fine_size = fine_model.serialize_params(f'remeshed/{args.mesh_name}/fine_weights_{args.hidden_dim}_{args.num_layers}.bin')/1024
    print(f"Compressed size: {coarse_size+fine_size:.2f} KB")

    try:
        f = open("results0.txt", "a")
        f.write(f"{args.mesh_name},{coarse_size+fine_size:.2f},{total_error:.2f}\n")
        f.close()
    except:
        f = open("results1.txt", "a")
        f.write(f"{args.mesh_name},{coarse_size+fine_size:.2f},{total_error:.2f}\n")
        f.close()

