# Overview  
Diffusion models (DMs) are a class of deep generative models that learn to produce data by gradually denoising a noisy sample. The process consists of two stages:

1. **Forward (diffusion) process** – a predefined Markov chain that adds noise to data, eventually turning it into pure Gaussian noise.  
2. **Reverse (denoising) process** – a learned Markov chain that removes the noise step‑by‑step, reconstructing realistic samples.

Originally inspired by non‑equilibrium thermodynamics, diffusion models have become the dominant approach for high‑fidelity image, audio, video, and molecular generation.

---

# Historical Background  

| Year | Milestone |
|------|-----------|
| 2015 | *Deep Unsupervised Learning using Nonequilibrium Thermodynamics* (Sohl‑Dickstein et al.) introduced the diffusion probabilistic framework. |
| 2011‑2018 | Connections to score matching, denoising autoencoders, and stochastic differential equations (SDEs) were explored. |
| 2020 | *Denoising Diffusion Probabilistic Models* (Ho, Jain, Abbeel) popularized the approach with a simple training objective based on predicting added noise. |
| 2021 | Implicit samplers (DDIM), score‑based generative modeling (Song & Ermon), and classifier‑free guidance broadened the toolkit. |
| 2022‑2023 | Latent Diffusion Models (LDM), Stable Diffusion, Imagen, and DALL·E 2 demonstrated that diffusion can match or surpass GANs on large‑scale text‑to‑image tasks. |
| 2024 | Distilled diffusion, DPM‑Solver, and multi‑modal diffusion models push towards real‑time sampling and broader modality coverage. |

---

# Core Concepts  

## Forward Diffusion Process  
The forward process transforms a data point $x_0$ into a noisy latent $x_T$ through a fixed schedule of Gaussian perturbations:

$
q(x_t \mid x_{t-1}) = \mathcal{N}\bigl(x_t; \sqrt{1-\beta_t}\,x_{t-1}, \beta_t \mathbf{I}\bigr)
$

- $\beta_t$ is the *noise schedule* (often linear, cosine, or quadratic).  
- After $T$ steps, $x_T \approx \mathcal{N}(\mathbf{0},\mathbf{I})$.

## Reverse Diffusion Process  
The reverse chain is parameterized by a neural network $\epsilon_\theta$ (or directly by a mean/variance predictor):

$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\bigl(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)\bigr)
$

Training aims to make the reverse distribution match the true posterior $q(x_{t-1}\mid x_t,x_0)$.  

## Training Objective  
A common simplification (used by DDPM) is the *noise‑prediction loss*:

$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0,\epsilon, t}\Bigl\|\epsilon - \epsilon_\theta\bigl(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t\bigl)\Bigr\|^2
$

where $\epsilon \sim \mathcal{N}(\mathbf{0},\mathbf{I})$ and $\bar\alpha_t = \prod_{s=1}^t (1-\beta_s)$.

## Noise Schedules  
| Schedule | Formula (for $t\in[1,T]$) | Typical Use |
|----------|-----------------------------|-------------|
| Linear   | $\beta_t = \beta_{\text{min}} + \frac{t-1}{T-1}(\beta_{\text{max}}-\beta_{\text{min}})$ | Simple baseline |
| Cosine   | $\alpha_t = \frac{f(t/T)}{f(0)}$, $f(u)=\cos\bigl(\frac{u+0.008}{1.008}\pi/2\bigr)^2$ | Improved sample quality |
| Quadratic| $\beta_t = (\beta_{\text{min}}^{0.5}+ \frac{t-1}{T-1}(\beta_{\text{max}}^{0.5}-\beta_{\text{min}}^{0.5}))^2$ | Faster early denoising |

---

# Architectural Variants  

## UNet Backbone  
Most diffusion models employ a **UNet** encoder‑decoder with skip connections, allowing the network to jointly process global context and fine‑grained details across diffusion steps.

## Attention Mechanisms  
- **Self‑attention** (within the latent) captures long‑range dependencies.  
- **Cross‑attention** enables conditioning on external modalities (e.g., text, segmentation maps).

## Conditioning Strategies  

| Technique | Description |
|-----------|-------------|
| **Classifier Guidance** | Uses a separate classifier $p_\phi(y\mid x_t)$ to steer the reverse process toward a target class. |
| **Classifier‑Free Guidance** | Trains a single model with a conditional and unconditional branch; at inference time the two predictions are interpolated. |
| **Prompt‑Embedding Conditioning** | Text embeddings (CLIP‑text, T5, BERT) are injected via cross‑attention layers. |
| **Latent Conditioning** | Operates in a compressed latent space (e.g., LDM), reducing compute while still allowing rich conditioning. |

---

# Prominent Diffusion Model Families  

## Denoising Diffusion Probabilistic Models (DDPM)  
The original “noise‑prediction” formulation; requires ~1000 sampling steps for high quality.

## Denoising Diffusion Implicit Models (DDIM)  
Deterministic sampler that can generate decent samples in far fewer steps (≈50–250) by altering the reverse transition distribution.

## Score‑Based Generative Modeling (SGM)  
Learns the *score* $\nabla_{x_t}\log q(x_t)$ of each noisy distribution and implements sampling as a discretized SDE (e.g., Euler–Maruyama).

## Latent Diffusion Models (LDM)  
Encode data into a low‑dimensional latent (via a pretrained autoencoder) and run diffusion only on that latent. This yields massive speed & memory gains (e.g., Stable Diffusion).

## Other Notable Variants  

| Model | Key Idea | Release |
|-------|----------|---------|
| **DDIM** | Implicit sampling with deterministic trajectories | 2021 |
| **DPM‑Solver** | High‑order ODE solvers for >10× speedup | 2022 |
| **Guided Diffusion** | Classifier‑free guidance for text‑to‑image | 2021 |
| **Imagen** | Large‑scale text‑to‑image diffusion with cascading super‑resolution | 2022 |
| **Stable Diffusion** | LDM with CLIP text conditioning; open‑source | 2022 |
| **AudioLDM** | Diffusion in latent space for text‑to‑audio generation | 2023 |
| **DiffWave** | Waveform diffusion for speech synthesis | 2021 |

---

# Applications  

- **Image Synthesis** – Photorealistic art, concept art, style transfer.  
- **Inpainting & Outpainting** – Fill missing regions or expand canvas boundaries.  
- **Super‑Resolution** – SR3, Real‑ESRGAN‑diffusion achieve 4× upscaling.  
- **Video Generation** – Temporal diffusion pipelines (e.g., Imagen Video).  
- **Audio & Speech** – DiffWave, WaveGrad, AudioLDM synthesize high‑fidelity waveforms.  
- **Molecular & Protein Design** – Generate SMILES strings, protein structures, and drug candidates.  
- **3‑D Generation** – Diffusion on point clouds, meshes, and NeRF representations.  

---

# Comparison with Other Generative Paradigms  

| Aspect | GANs | VAEs | Autoregressive | Diffusion |
|-------|------|------|----------------|-----------|
| **Training Stability** | Often unstable (mode collapse) | Stable (ELBO) | Stable, but slow | Very stable (simple loss) |
| **Sample Quality** | High (when trained well) | Moderate | High for discrete data | State‑of‑the‑art for images/audio |
| **Mode Coverage** | Can miss modes | Good coverage | Excellent (exact likelihood) | Excellent (explicit likelihood bound) |
| **Inference Speed** | One forward pass | One forward pass | Sequential (slow) | Typically 50‑1000 steps (slow) |
| **Conditional Flexibility** | Requires architecture tweaks | Easy (latent concat) | Straightforward | Very flexible (cross‑attention, guidance) |

---

# Evaluation Metrics  

- **Fréchet Inception Distance (FID)** – Measures similarity of generated vs. real image statistics.  
- **Inception Score (IS)** – Assesses image diversity and recognizability.  
- **Precision/Recall for Generative Models** – Quantifies fidelity vs. coverage.  
- **CLIPScore / BLIPScore** – Evaluates text‑image alignment.  
- **Human Preference Studies** – A/B testing on platforms like MTurk.  

---

# Advantages  

- Simple, unified training objective (L2 loss on noise).  
- Strong theoretical grounding (variational bound, score matching).  
- No adversarial dynamics → easier debugging.  
- Flexible conditioning (text, class labels, segmentation masks, etc.).  

---

# Limitations  

- **Sampling Speed** – Hundreds of neural network evaluations needed per sample.  
- **Memory Footprint** – Large UNet with multiple attention layers can exceed GPU memory.  
- **Large‑Scale Training Costs** – Training stable models often requires billions of image‑text pairs and hundreds of GPU days.  

---

# Acceleration Techniques  

| Technique | Principle |
|-----------|-----------|
| **DDIM / DPM‑Solver** | Use deterministic or higher‑order ODE solvers to reduce steps. |
| **Knowledge Distillation** | Train a smaller “student” model to imitate a large teacher’s sampling trajectory. |
| **Latent Diffusion** | Perform diffusion in a compressed latent space (e.g., VAE latent). |
| **Efficient UNet Variants** | Replace heavy ResNet blocks with MobileNet, EfficientNet, or depthwise‑separable convolutions. |
| **Parallel Sampling** | Leverage GPU batch processing across timesteps (e.g., “progressive distillation”). |
| **Hybrid Models** | Combine diffusion with GAN refinement for final‑step speedup. |

---

# Open Challenges & Future Directions  

1. **Real‑Time Sampling** – Achieving <30 ms per image while preserving quality.  
2. **Fine‑Grained Controllability** – Editing specific attributes (pose, lighting) without retraining.  
3. **Unified Multi‑Modal Diffusion** – Joint generation of text, image, audio, and 3‑D in a single framework.  
4. **Theoretical Understanding** – Deeper insights into why diffusion models avoid mode collapse and how noise schedules affect expressivity.  
5. **Energy‑Efficient Training** – Reducing carbon footprint via sparse diffusion steps or curriculum learning.  

---

# Notable Open‑Source Implementations  

- **🤗 Diffusers** – A versatile library supporting DDPM, DDIM, LDM, and many pretrained checkpoints.  
  <https://github.com/huggingface/diffusers>  
- **Stable Diffusion** – Official repository for the flagship latent diffusion model.  
  <https://github.com/CompVis/stable-diffusion>  
- **OpenAI Guided Diffusion** – Code and models for classifier‑free guidance.  
  <https://github.com/openai/guided-diffusion>  
- **Denoising Diffusion Implicit Models** – PyTorch implementation of DDIM.  
  <https://github.com/zhoukyin/ddim>  

---

# Example: Sampling from Stable Diffusion with 🤗 Diffusers  

```python
 Install the library (if needed)
 pip install diffusers transformers accelerate torch torchvision

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

 Load a pretrained checkpoint (FP16 for speed)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    revision="fp16"
).to("cuda")

 Swap to a fast scheduler (DPMSolver, 25 steps)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "A cyberpunk city at night, ultra‑realistic, detailed, photorealistic"
 Guidance scale >1 encourages adherence to the prompt
image = pipe(
    prompt,
    height=512,
    width=512,
    num_inference_steps=25,       # far fewer steps than the default 50‑100
    guidance_scale=7.5,
    eta=0.0,                      # deterministic mode
).images[0]

 Save the result
image.save("cyberpunk_city.png")
print("Image saved as cyberpunk_city.png")
```

*Key points in the script:*  
- **DPMSolverMultistepScheduler** provides a 10‑15× speedup.  
- **guidance_scale** controls how strongly the model follows the textual prompt.  
- **FP16** reduces memory and improves throughput on modern GPUs.

---

# References  

| # | Citation |
|---|----------|
| 1 | Sohl‑Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. ICLR, 2015. |
| 2 | Ho, J., Jain, A., & Abbeel, P. *Denoising Diffusion Probabilistic Models*. NeurIPS, 2020. |
| 3 | Song, Y., & Ermon, S. *Score‑Based Generative Modeling through Stochastic Differential Equations*. ICLR, 2021. |
| 4 | Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. *High‑Resolution Image Synthesis with Latent Diffusion Models*. CVPR, 2022. |
| 5 | Dhariwal, P., & Nichol, A. *Diffusion Models Beat GANs on Image Synthesis*. NeurIPS, 2021. |
| 6 | Saharia, C., et al. *Imagen: Text‑to‑Image Diffusion Models*. arXiv:2205.11487, 2022. |
| 7 | Kim, W., et al. *Denoising Diffusion Implicit Models*. ICML, 2021. |
| 8 | Gu, S., et al. *DiffusionGPT: Knowledge Transfer via Diffusion Model Distillation*. arXiv:2403.05356, 2024. |
| 9 | Liu, M., et al. *DPM‑Solver: A Fast ODE Solver for Diffusion Probabilistic Models*. ICML, 2022. |
|10| Romac, A., et al. *AudioLDM: Text‑to‑Audio Generation with Latent Diffusion Models*. arXiv:2210.01738, 2022. |

---

*This document provides a concise yet comprehensive snapshot of diffusion models in AI, suitable for a wiki entry, tutorial reference, or quick‑start guide.*