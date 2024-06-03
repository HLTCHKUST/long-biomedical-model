# How Long Is Enough? Exploring the Optimal Intervals of Long-Range Clinical Note Language Modeling

## Overview

Large pre-trained language models (LMs) have been widely adopted in biomedical and clinical domains, introducing many powerful LMs such as bio-lm and BioELECTRA. However, the applicability of these methods to real clinical use cases is hindered, due to the limitation of pre-trained LMs in processing long textual data with thousands of words, which is a common length for a clinical note. In this work, we explore long-range adaptation from such LMs with Longformer, allowing the LMs to capture longer clinical notes context. We conduct experiments on three n2c2 challenges datasets and a longitudinal clinical dataset from Hong Kong Hospital Authority electronic health record (EHR) system to show the effectiveness and generalizability of this concept, achieving 10% F1-score improvement. Based on our experiments, we conclude that capturing a longer clinical note interval is beneficial to the model performance, but there are different cut-off intervals to achieve the optimal performance for different target variables.

## Code Provided
Our code repository provides the implementation of four n2c2 classification tasks, i.e., n2c2 2006, n2c2 2008, n2c2 2014, and n2c2 2018. Our code enable the use of  `BioELECTRA`, `bio-lm`, `Longformer`-adapted `BioELECTRA`, and `Longformer`-adapted `bio-lm` models.

#### Setup
To use this repo:
1. Install the dependency through `pip install -r requirements.txt`
2. Put the local dataset inside `data/n2c2_2006_smokers/*`, `data/n2c2_2008/*`, etc
3. Run the finetuning. on your terminal do:
    - `bash run_n2c2_2006_smokers.sh` to finetune for n2c2_2006_smokers
    - `bash run_n2c2_2008.sh` to finetune for n2c2_2008
    - `bash run_n2c2_2014_risk_factors.sh` to finetune for n2c2_2014_risk_factors
    - `bash run_n2c2_2018_track1.sh` to finetune for n2c2_2018_track1

#### Supported Tasks
- n2c2_2006_smokers
- n2c2_2008
- n2c2_2014_risk_factors
- n2c2_2018_track1

## Citation
```
@inproceedings{cahyawijaya-etal-2022-long,
    title = "How Long Is Enough? Exploring the Optimal Intervals of Long-Range Clinical Note Language Modeling",
    author = "Cahyawijaya, Samuel  and
      Wilie, Bryan  and
      Lovenia, Holy  and
      Zhong, Huan  and
      Zhong, MingQian  and
      Ip, Yuk-Yu Nancy  and
      Fung, Pascale",
    editor = "Lavelli, Alberto  and
      Holderness, Eben  and
      Jimeno Yepes, Antonio  and
      Minard, Anne-Lyse  and
      Pustejovsky, James  and
      Rinaldi, Fabio",
    booktitle = "Proceedings of the 13th International Workshop on Health Text Mining and Information Analysis (LOUHI)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.louhi-1.19",
    doi = "10.18653/v1/2022.louhi-1.19",
    pages = "160--172",
    abstract = "Large pre-trained language models (LMs) have been widely adopted in biomedical and clinical domains, introducing many powerful LMs such as bio-lm and BioELECTRA. However, the applicability of these methods to real clinical use cases is hindered, due to the limitation of pre-trained LMs in processing long textual data with thousands of words, which is a common length for a clinical note. In this work, we explore long-range adaptation from such LMs with Longformer, allowing the LMs to capture longer clinical notes context. We conduct experiments on three n2c2 challenges datasets and a longitudinal clinical dataset from Hong Kong Hospital Authority electronic health record (EHR) system to show the effectiveness and generalizability of this concept, achieving {\textasciitilde}10{\%} F1-score improvement. Based on our experiments, we conclude that capturing a longer clinical note interval is beneficial to the model performance, but there are different cut-off intervals to achieve the optimal performance for different target variables.",
}
```
