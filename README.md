# How Far Are LLMs from Solving Olympiad-Level Physics Problems?

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2608.25097-b31b1b)](https://arxiv.org/abs/2608.25097)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E)](https://huggingface.co/datasets/physelite/PhysElite)
![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-684FA3)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit)

<!--
## Overview

PhysElite is a bilingual, multimodal benchmark for evaluating Olympiad-level physics reasoning in large language models. It includes **10K+ problems**, with visual diagrams, Chinese-English solution derivations, and final answers.

The paper evaluates 18 open-source and closed-source multimodal large language models. The strongest evaluated model achieves **33.7% answer accuracy**, highlighting the challenge of advanced physics reasoning. Step-level evaluation further examines where models make mistakes during their solutions.
-->

## Leaderboard

**Answer** is final-answer accuracy (%); **Process** measures step-level derivation quality. Subject columns report answer accuracy (%): **Mech.** = Mechanics, **E&M** = Electromagnetism, **Mod.** = Modern Physics, **Therm.** = Thermodynamics, and **Opt.** = Optics. Bold values mark the best model score in each metric.

| Rank | Model | Answer ↑ | Process ↑ | Mech. | E&M | Mod. | Therm. | Opt. |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| — | Human baseline | 48.5 | 65.2 | 47.6 | 42.3 | 54.2 | 51.7 | 57.3 |
| 🥇 1 | Grok-4.2 † | **33.7** | 47.6 | **34.0** | **26.6** | **33.0** | **35.4** | **25.4** |
| 🥈 2 | Claude-Opus-4.6 | 28.1 | **49.6** | 27.2 | 20.9 | 24.7 | 33.1 | 21.3 |
| 🥉 3 | o3-mini †⋆ | 26.2 | 43.9 | 25.3 | 20.4 | 26.8 | 28.4 | 19.6 |
| 4 | GPT-5.2 † | 24.0 | 42.6 | 22.1 | 18.6 | 23.9 | 28.4 | 17.4 |
| 5 | Gemini-3-Pro † | 21.8 | 41.4 | 21.9 | 15.5 | 20.0 | 23.7 | 16.1 |
| 6 | Kimi-K2-Thinking † | 20.0 | 32.7 | 19.3 | 13.2 | 15.6 | 22.8 | 9.6 |
| 6 | Qwen3-VL-235B-A22B | 20.0 | 36.8 | 19.3 | 15.8 | 19.2 | 22.0 | 14.4 |
| 8 | Claude-Sonnet-4.5 | 18.5 | 33.6 | 17.5 | 14.7 | 18.5 | 23.7 | 12.6 |
| 9 | Gemini-2.5 † | 17.9 | 36.8 | 16.9 | 14.2 | 20.7 | 20.9 | 13.5 |
| 10 | DeepSeek-V3 ⋆ | 17.0 | 34.6 | 18.7 | 10.3 | 14.5 | 19.3 | 10.9 |
| 11 | Qwen3-VL-32B | 15.9 | 31.5 | 15.2 | 10.1 | 13.8 | 17.8 | 12.2 |
| 12 | Qwen-VL-Max | 14.3 | 30.7 | 14.6 | 8.5 | 12.0 | 16.5 | 9.6 |
| 13 | Qwen3-VL-8B | 11.6 | 23.2 | 12.2 | 5.4 | 10.9 | 12.6 | 6.1 |
| 14 | GPT-4o | 10.4 | 25.0 | 8.8 | 6.5 | 10.2 | 13.8 | 10.0 |
| 15 | Qwen2.5-VL-72B | 9.8 | 21.5 | 9.1 | 6.2 | 9.1 | 11.1 | 6.1 |
| 16 | Dolphin-Mistral-24B | 8.6 | 20.6 | 6.1 | 7.5 | 10.1 | 9.9 | 7.0 |
| 17 | LLaMA-3.1-70B ⋆ | 7.5 | 20.2 | 4.9 | 6.0 | 10.5 | 7.9 | 6.6 |
| 18 | Qwen2.5-VL-7B | 1.7 | 3.7 | 2.4 | 0.5 | 0.7 | 2.0 | 0.5 |

† Reasoning model evaluated with extended thinking. ⋆ Text-only model evaluated without diagram input.

## Accessing the dataset

The benchmark data is hosted on [Hugging Face](https://huggingface.co/datasets/physelite/PhysElite). Its dataset viewer shows the available fields and lets you inspect examples before downloading. You can load it with the Hugging Face `datasets` library:

```python
from datasets import load_dataset

physelite = load_dataset("physelite/PhysElite")
print(physelite)
```

## Citation

If you use PhysElite in your research, please cite:

```bibtex
@inproceedings{
anonymous2026physelite,
title={PhysElite:  How Far Are {LLM}s from Solving Olympiad-Level Physics Problems?},
author={Anonymous},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems Evaluations and Datasets Track},
year={2026},
url={https://openreview.net/forum?id=f3ceI8ijyu}
}
```
