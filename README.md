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


1. you can run the init script in the root of the repository to:
   1. get the model without the Journal-Finder task dataset (you will be able to run the example_mlm notebook):
    ```bash
      chmod +x init.sh
      ./init.sh --target model
    ```
    2. get the dataset without the ScilitBERT pre-trained model (you will not be able to run any of the notebook):
   ```bash
      chmod +x init.sh
      ./init.sh --target dataset
    ```
    3. get both the model and the dataset (you will be able to run both the masked_mlm and the fine_tuning_journal_finder notebooks)
    ```bash
      chmod +x init.sh
      ./init.sh --target both
    ```

2. Get access to a Jupyter environment

3. Install a [PyTorch](https://pytorch.org/) version addapted to your CUDA version. (or run it on CPU, it is a no go for fine-tuning).

4. Install dependencies in your python environment using [pip](https://pypi.org/project/pip/) or [anaconda](https://www.anaconda.com/)
```
pip install -r ./requirements.txt
```
## Masked token prediction

If you followed the getting started steps and used the init script to dwnload the model, you can now explore the notebook: [notebooks/example_mlm.ipynb](./notebooks/example_mlm.ipynb)

## Fine-Tuning on the Journal Finder task

+ A fine-tuning quick start notebook on the Journal Finder task is given: [fine tuning example](./notebooks/fine_tuning_journal_finder.ipynb)

The hyper-parameters can be managed in the fine-tune function found in [utils](./notebooks/utils.py).

The fine tuned models are stored in the results folder (to rerun an experiment change the output folder or delete the previous output folder content.)

A csv describing the model performances on the test set will be generated in the file /evaluation_results/journal_finder_output.csv the first row describes the f1-score the following rows describe the top-k macro averaged accuracies for k ranging from 1 to 10.

## Contribute
You can contribute to this work by:

  + Helping to make the model ready for [publication on the Hugging Face model base](https://huggingface.co/transformers/model_sharing.html).
  + Finding good hyper-paremeters for the fine-tuning on the Journal-Finder task.