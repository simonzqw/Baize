import argparse, numpy as np, scanpy as sc
from scipy.sparse import issparse

def dmean(x): return np.asarray(x.mean(axis=0)).ravel() if issparse(x) else np.asarray(x).mean(axis=0)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data_path', required=True); p.add_argument('--out_npz', required=True)
    p.add_argument('--split_col', default='split'); p.add_argument('--train_split_value', default='train'); p.add_argument('--species_col', default='species'); p.add_argument('--human_value', default='human')
    p.add_argument('--perturb_col', default='perturbation'); p.add_argument('--context_col', default='cell_context'); p.add_argument('--control_key', default='control'); p.add_argument('--atac_key', default='atac_feat')
    p.add_argument('--bootstrap_reps', type=int, default=128); p.add_argument('--sample_ctrl', type=int, default=256); p.add_argument('--sample_pert', type=int, default=256); args=p.parse_args()
    ad=sc.read_h5ad(args.data_path)
    obs=ad.obs
    m=(obs[args.split_col].astype(str).values==args.train_split_value)&(obs[args.species_col].astype(str).values==args.human_value)
    ad=ad[m]
    perts=sorted(set(ad.obs[args.perturb_col].astype(str))-{args.control_key})
    ctxs=sorted(set(ad.obs[args.context_col].astype(str)))
    gctrl=dmean(ad[ad.obs[args.perturb_col].astype(str).values==args.control_key].X)
    gsrc={p:dmean(ad[ad.obs[args.perturb_col].astype(str).values==p].X)-gctrl for p in perts}
    out={k:[] for k in ['control_mean','target_mean','true_delta','source_delta','residual_target','atac_mean','perturbation','context']}
    rng=np.random.default_rng(42)
    for c in ctxs:
        cidx=np.where(ad.obs[args.context_col].astype(str).values==c)[0]
        cad=ad[cidx]
        ctrl_idx=np.where(cad.obs[args.perturb_col].astype(str).values==args.control_key)[0]
        if len(ctrl_idx)==0: continue
        for p in perts:
            pidx=np.where(cad.obs[args.perturb_col].astype(str).values==p)[0]
            if len(pidx)==0: continue
            for _ in range(args.bootstrap_reps):
                ci=rng.choice(ctrl_idx, size=min(args.sample_ctrl, len(ctrl_idx)), replace=True)
                pi=rng.choice(pidx, size=min(args.sample_pert, len(pidx)), replace=True)
                cm=dmean(cad[ci].X); pm=dmean(cad[pi].X); td=pm-cm; sd=gsrc[p]; rt=td-sd
                am=np.asarray(cad[pi].obsm[args.atac_key]).mean(0) if args.atac_key in cad.obsm else np.zeros(256, dtype=np.float32)
                out['control_mean'].append(cm); out['target_mean'].append(pm); out['true_delta'].append(td); out['source_delta'].append(sd); out['residual_target'].append(rt); out['atac_mean'].append(am); out['perturbation'].append(p); out['context'].append(c)
    np.savez_compressed(args.out_npz, **{k:np.asarray(v) for k,v in out.items()}); print('saved', args.out_npz, 'n=', len(out['perturbation']))
if __name__=='__main__': main()
