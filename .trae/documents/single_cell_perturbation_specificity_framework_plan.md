# Development Plan for the Single-cell Perturbation Specificity Modeling Framework (scERso)

We will build a simple and efficient MLP discriminator that learns whether a perturbation is compatible with a given cellular background.

## 1. Environment and project initialization

- Create the base directory structure: `models/`, `utils/`, and `data/`.
- Implement `synthetic_data.py` to generate simulated highly variable gene expression matrices and perturbation annotations, allowing the complete workflow to run without real data.

## 2. Core modules

### Data layer

- Implement `PerturbationDataset` to package `(RNA-seq, perturbation, label)` tuples.
- Concatenate high-dimensional RNA features with perturbation vectors represented by embeddings or one-hot encodings.

### Model layer

- Implement `SpecificityMLP`:
  - Input layer: receives the fused feature vector.
  - Hidden layers: fully connected layers with BatchNorm and Dropout to reduce overfitting.
  - Output layer: applies a sigmoid activation and returns a matching probability.

### Training and validation layer

- Implement `train.py` with a complete training loop, early stopping, and monitoring of AUC and accuracy.

## 3. Validation and delivery

- Run tests on synthetic data to verify convergence and correct identification of specificity patterns.
- Provide detailed documentation explaining how to replace highly variable genes with large pretrained embeddings.

## Key technical considerations

- **Feature dimensionality reduction:** initially use highly variable genes, such as a 2,000-dimensional RNA input.
- **Positive-negative balance:** use negative sampling when constructing mismatched examples.
- **Extensibility:** keep the model interface compatible with embeddings produced by pretrained models such as scGPT.
