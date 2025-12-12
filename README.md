# FCRASH (coming soon)
Protecting Facial Privacy Against AIGC Models via Machine Unlearning

##### Table of contents
1. [Environment setup](#environment-setup)
2. [Dataset preparation](#dataset-preparation)
3. [How to run](#how-to-run)
4. [Contacts](#contacts)

# Official PyTorch implementation of "FCRASH: Protecting Facial Privacy Against AIGC Models via Machine Unlearning"
<div align="center">
    <img width="1000" alt="teaser" src="assets/Teaser.png"/>
</div>

> **Abstract**: The rapid rise of text-to-image (T2I) generative models, such as Stable Diffusion, has raised significant concerns over the misuse of personal images, particularly through unauthorized personalization. We introduced FCRASH, a novel defense method designed to protect facial privacy against misuse in AI-generated content(AIGC). FCRASH introduces imperceptible, face-aware perturbations into user photos to prevent unauthorized face synthesis by diffusion-based generative models such as DreamBooth. By targeting facial regions critical for identity recognition, FCRASH effectively disrupts identity learning without sacrificing image quality.

**TLDR**: A security booth safeguards your privacy against malicious threats by preventing DreamBooth from synthesizing photo-realistic images of the individual target.

## Environment setup

Our code relies on the [diffusers](https://github.com/huggingface/diffusers) library from Hugging Face 🤗 and the implementation of latent caching from [ShivamShrirao's diffusers fork](https://github.com/ShivamShrirao/diffusers).

Install dependencies:
```shell
cd FCRASH
conda create -n fcrash python=3.9  
conda activate fcrash  
pip install -r requirements.txt  
```

Pretrained checkpoints of different Stable Diffusion versions can be **downloaded** from provided links in the table below:
<table style="width:100%">
  <tr>
    <th>Version</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>2.1</td>
    <td><a href="https://huggingface.co/stabilityai/stable-diffusion-2-1-base">stable-diffusion-2-1-base</a></td>
  </tr>
  <tr>
    <td>1.5</td>
    <td><a href="https://huggingface.co/runwayml/stable-diffusion-v1-5">stable-diffusion-v1-5</a></td>
  </tr>
  <tr>
    <td>1.4</td>
    <td><a href="https://huggingface.co/CompVis/stable-diffusion-v1-4">stable-diffusion-v1-4</a></td>
  </tr>
</table>

Please put them in `./stable-diffusion/`. We use Stable Diffusion version 2.1 in all of our experiments.

> GPU allocation: All experiments are performed on a NVIDIA 80GB H800 GPU.
> You have **8 NVIDIA H800 GPUs**, each with **81,559 MiB ≈ 81.6 GB** of memory. So yes, your GPUs have **81 GB of VRAM each**, which is **plenty** for training large models like SD3.

## Dataset preparation
For simple and convenient testing, we provided a simple dataset of several identities in './data/' to run.

For each identity, there's 12 images evenly divided into 3 subsets, including the reference clean set (set A), the target projecting set (set B), and an extra clean set for uncontrolled setting experiments (set C). 

## How to run
To defense Stable Diffusion version 2.1 with the Anti-DreamBooth baseline, you can run
```bash
bash scripts/aspl.sh
```

To defense Stable Diffusion version 2.1 with the Error-minimizing diffusion attack algorithm (unaccelerated and accelerated version), you can run
```bash
bash scripts/aspl_min.sh
bash scripts/unl_acc.sh
```

To defense Stable Diffusion version 2.1 with the Unlearnable/Error-minimizing vae attack algorithm, you can run
```bash
bash scripts/vae_attack.sh
```

With face-aware mechanism: 
```bash
bash scripts/face_aware.sh 
bash scripts/face_aware_vae_attack.sh
```

The same running procedure is applied for other supported algorithms:
<table style="width:100%">
  <tr>
    <th>Algorithm</th>
    <th>Bash script</th>
  </tr>
  <tr>
    <td>No defense</td>
    <td>scripts/attack_with_ensemble_aspl.sh</td>
  </tr>
  <tr>
    <td>FSMG</td>
    <td>scripts/attack_with_fsmg.sh</td>
  </tr>
  <tr>
    <td>T-FSMG</td>
    <td>scripts/attack_with_targeted_fsmg.sh</td>
  </tr>
  <tr>
    <td>E-FSMG</td>
    <td>scripts/attack_with_ensemble_fsmg.sh</td>
  </tr>
</table>

If you want to train a DreamBooth model from your own data, whether it is clean or perturbed, you may run the following script:
```
bash scripts/train_dreambooth_alone.sh
```

Inference: generates examples with multiple-prompts
```
python infer.py --model_path <path to DREAMBOOTH model>/checkpoint-1000 --output_dir ./test-infer/
```

## Limitations
The picture generated do not successfully learn the concept of the "sks" person, making us couldn't really determine if the attack is really successful or not. 

## Contacts
Email: [sylviachung.22@intl.zju.edu.cn](mailto:sylviachung.22@intl.zju.edu.cn).
