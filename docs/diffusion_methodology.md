# Current scERso Diffusion Methodology

## 1. Task definition and conditional modeling

The model learns the conditional distribution:

\[
p(\mathbf{x}_{\text{pert}}\mid \mathbf{x}_{\text{ctrl}},\ \text{perturb},\ \text{cell\_line},\ \text{dose},\ \text{ATAC},\ \text{drug})
\]

Here, \(\mathbf{x}_{\text{ctrl}}\) is the control RNA-expression profile and \(\mathbf{x}_{\text{pert}}\) is the post-perturbation expression profile.

## 2. Semantic latent variables and context

The model first encodes multimodal conditions into a semantic latent variable \(\mathbf{z}_{sem}\):

- projection of the control RNA profile;
- perturbation embedding, optionally scaled by dose;
- cell-line embedding;
- dose projection;
- optional ATAC and drug-feature projections;
- multi-head self-attention followed by a residual MLP and LayerNorm to produce \(\mathbf{z}_{sem}\);
- a parallel joint semantic encoder that concatenates RNA, perturbation, cell-line, and dose features, processes them with an MLP, and combines them with the attention pathway through a gate to stabilize single-perturbation representations.

The diffusion condition vector is then constructed as:

\[
\mathbf{c}=\left[\mathbf{x}_{\text{ctrl}};\mathbf{z}_{sem}\right]
\]

Conditional dropout is supported during training by randomly setting \(\mathbf{z}_{sem}\) to zero, enabling classifier-free guidance.

## 3. Forward diffusion process

A Gaussian forward diffusion process is used:

\[
q(\mathbf{x}_t\mid \mathbf{x}_0)=\mathcal{N}\left(\sqrt{\bar\alpha_t}\mathbf{x}_0,(1-\bar\alpha_t)\mathbf{I}\right)
\]

It is implemented as:

\[
\mathbf{x}_t=\sqrt{\bar\alpha_t}\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,\quad \boldsymbol\epsilon\sim\mathcal{N}(0,\mathbf{I})
\]

The noise schedule can be linear or cosine, with cosine used by default.

## 4. Reverse denoising network

The denoiser is a Squidiff-style MLP:

- it receives \(\mathbf{x}_t\) and the condition \(\mathbf{c}\);
- timestep \(t\) is represented using sinusoidal positional encoding followed by an MLP;
- the time embedding and \(\mathbf{z}_{sem}\) are injected into every residual block;
- the output has the same dimensionality as the expression input.

The current objective is `pred_x0`, meaning that the network directly predicts \(\hat{\mathbf{x}}_0\).

## 5. Training objective

At every iteration, a timestep \(t\) is sampled and the following objective is minimized:

\[
\mathcal{L}=\mathbb{E}_{t,\mathbf{x}_0,\boldsymbol\epsilon}\left[\lVert f_\theta(\mathbf{x}_t,t,\mathbf{c})-\mathbf{x}_0\rVert_2^2\right]
\]

The implementation first averages the loss over features for each sample and then averages over the batch. Optional sample weights can be supplied by the timestep resampler.

## 6. Sampling and inference

### DDPM sampling

The model iterates from \(t=T-1\) to \(0\):

1. predict \(\hat{\mathbf{x}}_0\), or predict noise and convert it to \(\hat{\mathbf{x}}_0\);
2. sample \(\mathbf{x}_{t-1}\) from the posterior mean and variance of \(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\hat{\mathbf{x}}_0)\).

### Fast DDIM sampling

When `sample_steps < timesteps`, a DDIM subsequence is used. The parameter \(\eta\) controls stochasticity.

### Latent interpolation

A linear interpolation trajectory can be constructed between two semantic latent vectors:

\[
z(\alpha)=(1-\alpha)z_A+\alpha z_B,\quad \alpha\in[0,1]
\]

This can be used to analyze continuous dose or state transitions with `predict_diffusion.py --interpolate_to --interp_steps`.

### Classifier-free guidance

Conditional and unconditional predictions are combined as:

\[
\hat{y}=\hat{y}_{uncond}+s(\hat{y}_{cond}-\hat{y}_{uncond})
\]

where \(s\) is `guidance_scale`.

## 7. Mathematical interpretation

1. **High-dimensional expression generation is converted into progressive refinement.** The reverse process starts from isotropic Gaussian noise and gradually contracts toward an expression vector consistent with the conditional distribution.
2. **The conditional latent variable \(\mathbf{z}_{sem}\) acts as a perturbation-semantic coordinate.** Perturbation, cell-line, dose, ATAC, and drug information are represented in one latent space and shape the reverse diffusion trajectory.
3. **The `pred_x0` objective directly supervises the biological signal.** Compared with pure noise prediction, direct supervision of \(\mathbf{x}_0\) provides a more direct fit to expression amplitudes, although it relies on appropriate normalization and calibration.
4. **Classifier-free guidance amplifies the conditional contribution.** It improves condition consistency during sampling, usually at the cost of some diversity.
5. **Timestep resampling follows an importance-sampling principle.** Loss-second-moment sampling focuses on high-loss timesteps, approximately reducing gradient variance and improving sample efficiency.

## 8. Relationship to combinatorial perturbations

The implementation can encode individual perturbation latent vectors, combine them, and then sample the resulting response:

- `sum/mean`: linear composition that is stable and directly interpretable;
- `adaptive`: weighted linear composition augmented with pairwise nonlinear interactions,
  \(\phi([z_i,z_j,z_i\odot z_j,|z_i-z_j|])\), followed by gated fusion:

  \[
  z_{combo}=g\odot z_{lin} + (1-g)\odot (z_{lin}+z_{pair})
  \]

  where \(g=\sigma(\psi([z_{lin},\bar z]))\).

This formulation captures part of the synergistic or antagonistic nonlinear response while preserving the original single-perturbation pathway.
