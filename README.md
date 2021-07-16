<div><img src="images/mdpi_logo.jpg" alt="drawing" width="200"/><img src="images/scilit_logo.jpeg" alt="drawing" width="200"/></div>

# ScilitBERT Example
In this repository, you will find:
+ A [Jupyter](https://jupyter.org/) notebook explaining how to load ScilitBERT and its tokenizer using [Hugging Face](https://huggingface.co/)
+ How to test the Mask filling feature 
+ A dataset for fine-tuning on the Journal Finder task
+ A notebook to quick start the fine-tuning.

## What is ScilitBERT?

ScilitBERT is a BERT model for academic language representation developed by [MDPI](https://www.mdpi.com/). The training data is extracted from [Scilit](https://www.scilit.net/). for more details check the paper available at: (not available at the moment)

## Getting started

1. Download a zip file containing ScilitBERT pretrained and its tokenizer at the link below: [res.mdpi.com/data/ScilitBERT_plus_tokenizer.zip](res.mdpi.com/data/ScilitBERT_plus_tokenizer.zip)

To run the notebook you can unzip the downloaded file in the root of this repository.

2. Get access to a Jupyter environment

3. Install a [PyTorch](https://pytorch.org/) version addapted to your CUDA version. (or run it on CPU, it is a no go for fine-tuning).

4. Install dependencies in your python environment using [pip](https://pypi.org/project/pip/) or [anaconda](https://www.anaconda.com/)
```
pip install -r ./requirements.txt
```

4. You can now run the notebook: [notebooks/example_mlm.ipynb](./notebooks/example_mlm.ipynb)

## Fine-Tuning on the Journal Finder task

+ A fine-tuning quick start notebook on the Journal Finder task is given:
+ We add a basic fine-tuning example notebook: [fine tuning example](./notebooks/fine_tuning_journal_finder.ipynb)
  
The hyper-parameters can be managed in the fine-tune function found in [utils](./notebooks/utils.py).

## Contribute
You can contribute to this work by:

  + Helping to make the model ready for [publication on the Hugging Face model base](https://huggingface.co/transformers/model_sharing.html).
  + Finding good hyper-paremeters for the fine-tuning on the Journal-Finder task.

