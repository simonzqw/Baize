# scERso Conditional Diffusion Model for Single-cell Response Prediction

The repository's **main workflow now uses the conditional diffusion model**. The legacy MLP/Transformer training path (`train.py`) is retained for historical comparison but is no longer recommended.

Use the following entry points for current experiments:
- Training: `train_diffusion.py`
- Evaluation: `evaluate_diffusion.py`
- Inference, perturbation composition, and interpolation: `predict_diffusion.py`
- Visualization: `visualize_diffusion.py`

---

## 1. Main capabilities

### 1.1 Conditional diffusion backbone
- Background-effect disentanglement: `z_bg` (background) and `z_eff` (effect) are encoded separately.
- Target modes: `target_mode = target | delta`.
- Sampling and guidance: classifier-free guidance, configurable DDIM sampling steps, and EMA-weight evaluation are supported.

### 1.2 Multiple task modes
Use `--task_mode` to select the task definition and avoid mismatches between the task and the input data:

1. `single_gene`
   - Designed for single-gene perturbation datasets such as Adamson.
   - Condition fields: `perturb_gene_idx` and `is_control`.

2. `translation`
   - Designed for two-condition translation tasks, such as day4 -> day6.
   - Condition fields: `condition_id` and `source_flag`.

3. `drug`
   - Designed for drug-response prediction with optional molecular structure, dose, and cellular-context conditions.

### 1.3 Data and control/reference handling
- Supports `split_strategy = random | perturbation | custom`.
- Under perturbation zero-shot splits, validation and test samples reuse the training control bank when their own splits contain no controls.
- Supports `control_match_mode`, `control_prototype_mode`, and `control_prototype_temp`.

---

## 2. Project structure

- `train_diffusion.py`: primary conditional-diffusion training entry point.
- `evaluate_diffusion.py`: single-cell and perturbation-level evaluation.
- `predict_diffusion.py`: single and combinatorial perturbation prediction and latent interpolation.
- `visualize_diffusion.py`: combinatorial perturbation analysis and diagnostic visualization.
- `models/scerso_diffusion.py`: conditional diffusion model definition.
- `models/diffusion_core.py`: diffusion process implementation.
- `utils/data_processor.py`: h5ad loading, split construction, control pools, and condition fields.
- `docs/diffusion_methodology.md`: methodological description.

> Legacy files such as `train.py`, `evaluate_metrics.py`, and `visualize.py` are retained only for historical comparison and are not part of the recommended workflow.

---

## 3. Environment and dependencies

Recommended environment:
- Python 3.8+
- PyTorch
- scanpy / anndata
- numpy / scipy / pandas
- scikit-learn
- matplotlib / seaborn
- rdkit, only when SMILES-derived drug features are used

The following setting is also recommended:

```bash
export OMP_NUM_THREADS=1
```

---

## 4. Training

### 4.1 Adamson (`single_gene`)

```bash
python train_diffusion.py \
  --data_path /path/to/adamson/perturb_processed.h5ad \
  --save_dir ./checkpoints_adamson_single_gene \
  --task_mode single_gene \
  --split_strategy perturbation \
  --preset vnext \
  --amp
```

When the training data contain combinatorial labels such as `double_...`, `triple_...`, or `GENE1+GENE2+GENE3`, enable multi-gene label parsing:

```bash
python train_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --save_dir ./checkpoints_combo_diffusion \
  --task_mode single_gene \
  --split_strategy perturbation \
  --perturb_parse_mode multi_gene_parse \
  --preset vnext \
  --amp
```

### 4.2 day4/day6 (`translation`)

```bash
python train_diffusion.py \
  --data_path /path/to/day4_to_day6_diffusion.h5ad \
  --save_dir ./checkpoints_day4_day6_translation \
  --task_mode translation \
  --split_strategy custom \
  --split_col split \
  --atac_key atac_feat \
  --preset vnext \
  --amp
```

> Use `--preset smoke` for a quick smoke test.

---

## 5. Evaluation

```bash
python evaluate_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --task_mode single_gene \
  --split_strategy perturbation \
  --output_json ./checkpoints_xxx/eval_metrics.json
```

A three-gene composition case can also be evaluated. When the corresponding combination label is present in the h5ad file, the output includes metrics against its observed mean response:

```bash
python evaluate_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --task_mode single_gene \
  --split_strategy perturbation \
  --perturb_parse_mode multi_gene_parse \
  --cell_line K562 \
  --combo_genes FOXA2 GATA6 SOX17 \
  --latent_mode adaptive \
  --output_json ./checkpoints_xxx/eval_metrics_triple.json
```

For translation data, use:

```bash
--task_mode translation --split_strategy custom --split_col split
```

---

## 6. Inference and visualization

### 6.1 Prediction, composition, and interpolation

```bash
python predict_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --cell_line K562 \
  --perturb_genes FOXA2 GATA6 \
  --latent_mode adaptive \
  --save_dir ./pred_out
```

For a three-gene perturbation, append the third gene to the original two-gene command:

```bash
python predict_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --cell_line K562 \
  --perturb_genes FOXA2 GATA6 SOX17 \
  --latent_mode adaptive \
  --save_dir ./pred_out_triple
```

### 6.2 Visualization

```bash
python visualize_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --cell_line K562 \
  --perturb_genes FOXA2 GATA6 \
  --save_path ./combo_report.png
```

Three-gene composition visualization:

```bash
python visualize_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --cell_line K562 \
  --perturb_genes FOXA2 GATA6 SOX17 \
  --latent_mode adaptive \
  --save_path ./triple_combo_report.png
```

---

## 7. Frequently asked questions

### Q1: `adata.obs is missing the custom split column: split`
The command uses `split_strategy=custom`, but the data do not contain `obs['split']`. Either use:

```bash
--split_strategy perturbation
```

or create the `split` column in the h5ad file before training.

### Q2: I passed `--split_strategy perturbation`, but the log still reports `custom`
`--preset` only overrides parameters that were not explicitly supplied. Explicit command-line arguments are retained. If the issue persists, check whether the same argument appears more than once in the command.

### Q3: The validation or test split reports an empty control pool
Under a perturbation zero-shot split, validation and test data reuse the training control bank. If the error persists, the training split itself probably contains no control samples and the source data should be checked.

---

## 8. Cross-species perturbation prediction module

This module is independent from the diffusion mainline.

### 8.1 Legacy scripts

- `scripts/prepare_mouse_context.py`
- `scripts/train_cross_species_ctx.py`
- `scripts/cross_species_infer_ctx.py`

### 8.2 Recommended v2 workflow

1. Diagnose data: `scripts/cross_species_diagnose_data.py`
2. Build bootstrap pseudo-bulk: `scripts/cross_species_build_pseudobulk.py`
3. Train residual model: `scripts/cross_species_train_residual.py`
4. Run context-wise inference: `scripts/cross_species_infer_residual.py`
5. Evaluate: `scripts/evaluate_cross_species_mouse.py`, `scripts/evaluate_cross_species_context_preds.py`
