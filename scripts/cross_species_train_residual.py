import argparse, os, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from models.cross_species_residual import CrossSpeciesResidualPredictor
from models.cross_species_losses import weighted_delta_mse, topk_corr_loss, sign_consistency_loss, delta_norm_loss, residual_l2_loss

def main():
    p=argparse.ArgumentParser(); p.add_argument('--pseudobulk_npz', required=True); p.add_argument('--save_dir', required=True); p.add_argument('--perturb_vocab_path', required=True)
    p.add_argument('--use_atac', action='store_true'); p.add_argument('--rank', type=int, default=64); p.add_argument('--hidden_dim', type=int, default=512); p.add_argument('--residual_scale', type=float, default=0.01)
    p.add_argument('--epochs', type=int, default=100); p.add_argument('--batch_size', type=int, default=64); p.add_argument('--lr', type=float, default=3e-5); p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--delta_mse_weight', type=float, default=1.0); p.add_argument('--topk_corr_weight', type=float, default=0.5); p.add_argument('--sign_weight', type=float, default=0.2); p.add_argument('--norm_weight', type=float, default=0.1); p.add_argument('--correction_l2_weight', type=float, default=0.01)
    args=p.parse_args(); os.makedirs(args.save_dir, exist_ok=True)
    d=np.load(args.pseudobulk_npz, allow_pickle=True)
    with open(args.perturb_vocab_path) as f: vocab=[x.strip() for x in f if x.strip()]
    p2i={p:i for i,p in enumerate(vocab)}
    pid=np.array([p2i.get(x,0) for x in d['perturbation'].astype(str)], dtype=np.int64)
    ds=TensorDataset(torch.tensor(d['control_mean']).float(), torch.tensor(d['source_delta']).float(), torch.tensor(d['atac_mean']).float(), torch.tensor(pid).long(), torch.tensor(d['target_mean']).float(), torch.tensor(d['true_delta']).float())
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True)
    model=CrossSpeciesResidualPredictor(n_genes=ds.tensors[0].shape[1], n_perturbations=len(vocab), use_atac=args.use_atac, rank=args.rank, hidden_dim=args.hidden_dim, residual_scale=args.residual_scale)
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best=1e9
    for ep in range(args.epochs):
        model.train(); tot=0.0
        for c,s,a,p,t,td in dl:
            o=model(c,s,p,a if args.use_atac else None)
            loss=args.delta_mse_weight*weighted_delta_mse(o['pred_delta'],td)+args.topk_corr_weight*topk_corr_loss(o['pred_delta'],td)+args.sign_weight*sign_consistency_loss(o['pred_delta'],td)+args.norm_weight*delta_norm_loss(o['pred_delta'],td)+args.correction_l2_weight*residual_l2_loss(o['correction'])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()*c.shape[0]
        avg=tot/len(ds)
        if avg<best:
            best=avg; torch.save({'model':model.state_dict(),'vocab':vocab}, os.path.join(args.save_dir,'best_model_cross_species_residual.pth'))
        if ep%10==0: print(ep,avg)
    print('best_loss',best)
if __name__=='__main__': main()
