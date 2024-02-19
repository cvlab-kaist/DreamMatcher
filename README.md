# DreamMatcher: Appearance Matching Self-Attention for Semantically-Consistent Text-to-Image Personalization

This is the official implementation of the paper "DreamMatcher: Appearance Matching Self-Attention
for Semantically-Consistent Text-to-Image Personalization" by Jisu Nam, Heesu Kim, DongJae Lee, Siyoon Jin, Seungryong Kim†, and Seunggyu Chang†.

![Teaser](./images/teaser.png)

For more information, check out the [[project page](https://ku-cvlab.github.io/DreamMatcher/)].

# Environment Settings

```
git clone https://github.com/KU-CVLAB/DreamMatcher.git
cd DreamMatcher

conda env create -f environment.yml
conda activate dreammatcher
pip install -r requirements.txt

cd diffusers
pip install -e .
```

# Pre-trained Weights

You can run DreamMatcher with any off-the-shelf personalized models. We provide pre-trained personalized models of [Textual Inversion](https://github.com/rinongal/textual_inversion), [DreamBooth](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion), and [CustomDiffusion](https://github.com/adobe-research/custom-diffusion) on the [ViCo](https://github.com/haoosz/ViCo) dataset. You can find the pre-trained weights on [Link](https://drive.google.com/drive/folders/1R2-x8BpqryVTjYBKmafCA6YJ0cLtNi1s?usp=share_link).

# Dataset

## Image Dataset

For a fair and unbiased evaluation, we used the [ViCo](https://github.com/haoosz/ViCo) image and prompt dataset gathered from Textual Inversion, DreamBooth, and CustomDiffusion. The dataset comprises 16 concepts, including 5 live objects and 11 non-live objects. In the _./inputs_ folder, you can see 4-12 images of each of the 16 concepts.

## Prompt Dataset

We provide the [ViCo](https://github.com/haoosz/ViCo) prompt dataset for live objects in _./inputs/prompts_live_objects.txt_ and for non-live objects in _./inputs/prompts_nonlive_objects.txt_.

Additionally, for evaluation in more complex scenarios, we propose the challenging prompt dataset, which is available in _./inputs/prompts_live_objects_challenging.txt_ and _./inputs/prompts_nonlive_objects_challenging.txt_.

# Run DreamMatcher

To run DreamMatcher, select a personalized model from "ti", "dreambooth", or "custom_diffusion" as the baseline. Below, we provide example code using "dreambooth" as the baseline, with 8 samples. Output images for both the baseline and DreamMatcher will be saved in the result directory.

Run DreamMatcher on the ViCo prompt dataset:

      python run_dreammatcher.py --models "dreambooth" --result_dir "./results/dreambooth/test" --num_samples 8 --num_device 0 --mode "normal"

Run DreamMatcher on the proposed challenging prompt dataset:

      python run_dreammatcher.py --models "dreambooth" --result_dir "./results/dreambooth/test" --num_samples 8 --num_device 0 --mode "challenging"

# Evaluation

To evaluate DreamMatcher, specify the result directory containing the result images from both the baseline and DreamMatcher. I<sub>CLIP</sub>, I<sub>DINO</sub>, and T<sub>CLIP</sub> metrics will be calculated.

Evaluation on the ViCo prompt dataset :

      python evaluate.py --result_dir "./results/dreambooth/test" --mode "normal"

Evaluation on the proposed challenging prompt dataset :

      python evaluate.py --result_dir "./results/dreambooth/test" --mode "challenging"

# Collect Evaluation Results

Collect evaluation results for every concept in the result directory:

      python collect_results.py --result_dir "./results/dreambooth/test"

# Results

### Qualitative comparision with baselines for live objects:

![Base_live](./images/base_live.png)

### Qualitative comparision with baselines for non-live objects:

![Base_nonlive](./images/base_nonlive.png)

### Qualitative comparison with previous works for live objects:

![Sota_live](./images/sota_live.png)

### Qualitative comparison with previous works for non-live objects:

![Sota_nonlive](./images/sota_nonlive.png)

# Acknowledgement <a name="Acknowledgement"></a>

We have mainly borrowed code from the public project [MasaCtrl](https://github.com/TencentARC/MasaCtrl). A huge thank you to the authors for their valuable contributions.

# BibTeX

If you find this research useful, please consider citing:

```BibTeX
@misc{nam2024dreammatcher,
      title={DreamMatcher: Appearance Matching Self-Attention for Semantically-Consistent Text-to-Image Personalization}, 
      author={Jisu Nam and Heesu Kim and DongJae Lee and Siyoon Jin and Seungryong Kim and Seunggyu Chang},
      year={2024},
      eprint={2402.09812},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
