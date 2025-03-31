# POYO: A Unified, Scalable Framework for Neural Population Decoding

POYO (Azabou et al 2023, NeurIPS) introduces a new transformer-based framework for neural population decoding, designed to adapt rapidly to new, unseen sessions with minimal labels, leveraging large-scale neural recordings. Read here for a [high-level intro to POYO](https://poyo-brain.github.io/).

The code for POYO has been moved to the [torch_brain](https://github.com/neuro-galaxy/torch_brain) package. You can find the code under `examples/poyo`.

### Installation

```bash
git clone https://github.com/neuro-galaxy/torch_brain.git
cd torch_brain/examples/poyo
````

### Training POYO-MP
To train POYO-MP you first need to download the [`perich_miller_population_2018`](https://brainsets.readthedocs.io/en/latest/glossary/brainsets.html#perich-miller-population-2018) data using [brainsets](https://github.com/neuro-galaxy/brainsets).

```bash
brainsets prepare perich_miller_population_2018
```

Then you can train POYO-MP by running:

```bash
python train.py --config-name train_poyo_mp.yaml
```

Checkout `configs/base.yaml` and `configs/train_poyo_mp.yaml` for all configurations available.

### Training POYO-1
To train POYO-1 you first need to download all datasets using `brainsets`.

```bash
brainsets prepare perich_miller_population_2018
brainsets prepare churchland_shenoy_neural_2012
brainsets prepare flint_slutzky_accurate_2012
brainsets prepare odoherty_sabes_nonhuman_2017
```

Then you can train POYO-1 by running:

```bash
python train.py --config-name train_poyo_1.yaml
```

### Pre-trained Weights

Below is a table of pre-trained weights for POYO models that you can download and use:

| Model | Description | Link |
|-------|-------------|------|
| POYO-MP | Trained on the perich_miller_population_2018 dataset | [Download](https://torch-brain.s3.amazonaws.com/model-zoo/poyo_mp.ckpt) |
| POYO-1 | Trained on all four datasets | Coming soon |

## Cite
Please cite [our paper](https://papers.nips.cc/paper_files/paper/2023/hash/8ca113d122584f12a6727341aaf58887-Abstract-Conference.html) if you use this code in your own work:

```bibtex
@inproceedings{
    azabou2023unified,
    title={A Unified, Scalable Framework for Neural Population Decoding},
    author={Mehdi Azabou and Vinam Arora and Venkataramana Ganesh and Ximeng Mao and Santosh Nachimuthu and Michael Mendelson and Blake Richards and Matthew Perich and Guillaume Lajoie and Eva L. Dyer},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```
