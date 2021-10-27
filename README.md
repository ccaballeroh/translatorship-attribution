# On Explainable Features for Translatorship Attribution: Unveiling the Translator's Style with Causality

## Description
Code for reproducing results presented in the paper [_On Explainable Features for Translatorship Attribution: Unveiling the Translator's Style with Causality_](https://ieeexplore.ieee.org/document/9467290) published in IEEE Access, an Open Access Journal, on June 29, 2021.

Cite as:

```
@ARTICLE{9467290,
author={Caballero, Christian and Calvo, Hiram and Batyrshin, Ildar},
journal={IEEE Access},
title={On Explainable Features for Translatorship Attribution: Unveiling the Translator’s Style With Causality},
year={2021},
volume={9},
number={},
pages={93195-93208},
doi={10.1109/ACCESS.2021.3093370}}
```

## Instructions

The results can be reproduced in two ways: locally and on the cloud.

### Locally

For this, you must have Anaconda (or at least miniconda) installed.

1. Download or clone this repository locally.
1. Open a conda prompt inside the directory and type the following command:
    ``` bash
    $ conda env create -n translator-attribution -f environment.yml
    ```   
    This will create a new environment called `translator-attibution` with the exact versions of the libraries used to produce the results.
1. From any Anaconda environment, open `jupyter lab` and you can start running the Notebooks in order (select the recently created kernel&mdash;`translator-attribution`).

### Remotely on Colab

For this other option, you need a Google Drive account.
    
1. Download the contents of this repository in a folder named `translator-attribution` in your Google Drive root folder.
1. You can start running the Notebooks in Colab from this repository (allow Drive to synch between steps) by clicking on the Colab badge on the top of each notebook.

    |  |              Notebook                  | Description |
    |:--|:--------------------------------------|:------------|
    |1| [Processing](./01Processing.ipynb) | Preprocessing and processing of text files |
    |2| [Feature Extraction](./02Feature_Extraction.ipynb)| Extraction of features and saving to disk in JSON format |
    |3| [Experiments](./03Experiments.ipynb) | All experiments and the results are saved to disk |
    |4| [Most Relevant Features](./04Extraction_Most_Relevant_Features.ipynb) | Extraction and saving to disk of most relevant features ("translator fingerprints") for each translator and for each feature set |

## Results

The results are saved as LaTeX tables and SVG images on the [results](./results) folder. They are under version control and can be browsed on this repository. 
