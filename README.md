<div align="center">

<h1>Bokeh Diffusion: Defocus Blur Control in Text-to-Image Diffusion Models</h1>

**[Armando Fortes](https://atfortes.github.io/) &emsp; [Tianyi Wei](https://scholar.google.com/citations?user=-wfXmM4AAAAJ&hl=en) &emsp; [Shangchen Zhou](https://shangchenzhou.com) &emsp; [Xingang Pan](https://xingangpan.github.io)**

S-lab, Nanyang Technological University  

<div>
    <h4 align="center">
        <a href="https://atfortes.github.io/projects/bokeh-diffusion/" target='_blank'>
        <img src="https://img.shields.io/badge/🏠-Project%20Page-blue">
        </a>
        <a href="https://arxiv.org/abs/2503.08434" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2503.08434-b31b1b.svg">
        </a>
<!--         <img src="https://visitor-badge.glitch.me/badge?page_id=atfortes.BokehDiffusion" alt="visitor badge"> -->
    </h4>
</div>

**Bokeh Diffusion enables precise, scene-consistent bokeh transitions in text-to-image diffusion models**

![teaser](./assets/teaser.png)

🎥 For more visual results, check out our <a href="https://atfortes.github.io/projects/bokeh-diffusion/" target="_blank">project page</a>

🚀✨🚧 We are working hard on releasing the code... 🔧🛠️💻 Stay tuned! 🚧✨🚀

</div>

## 📮 Update
- [2025.03] This repo is created.

## 🚧 TODO
- [ ] Release Dataset
- [ ] Release Model Weights
- [ ] Release Inference Code
- [ ] Release Training Code
- [ ] Release FLUX Version
- [ ] Release HuggingFace Demo

## 🔎 Overview

Bokeh Diffusion combines three key components to produce lens-like bokeh without altering the generated scene content:

**(1) Hybrid Dataset Pipeline:** We merge real in-the-wild images with synthetic bokeh augmentations. This approach ensures realism and diversity while also providing contrastive examples for training.

![dataset](./assets/dataset.png)

**(2) Defocus Blur Conditioning:** A physically interpretable defocus blur parameter is injected via decoupled cross-attention.

**(3) Grounded Self-Attention:** A *pivot* image anchors the generated scene, ensuring content preservation across bokeh levels.

![method](./assets/method.png)

## 📑 Citation

If you find our work useful, please cite the following paper:

```bibtex
@article{fortes2025bokeh,
    title     = {Bokeh Diffusion: Defocus Blur Control in Text-to-Image Diffusion Models},
    author    = {Fortes, Armando and Wei, Tianyi and Zhou, Shangchen and Pan, Xingang},
    journal   = {arXiv preprint arXiv:2503.08434},
    year      = {2025},
}
```

## ©️ License

This project is licensed under [NTU S-Lab License 1.0](https://github.com/atfortes/BokehDiffusion/blob/main/LICENSE). Redistribution and use should follow this license.

## 🤝 Acknowledgements

We would like to thank the following projects that made this work possible:

- [Megalith-10M](https://huggingface.co/datasets/madebyollin/megalith-10m) is used as the base dataset for collecting real in-the-wild photographs.
- [BokehMe](https://github.com/lucidrains/diffusion-models) provides the synthetic blur rendering engine for generating defocus augmentations.
- [Depth-Pro](https://github.com/apple/ml-depth-pro) is used to estimate metric depth maps.
- [RMBG v2.0](https://huggingface.co/briaai/RMBG-2.0) is used to generate foreground masks.
- [FLUX](https://github.com/black-forest-labs/flux) & [Realistic-Vision](https://huggingface.co/SG161222/Realistic\_Vision\_V5.1\_noVAE) & [Cyber-Realistic](https://huggingface.co/cyberdelia/CyberRealistic) are used as the base models for generating the samples in the paper.
