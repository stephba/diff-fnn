# DiffFNN: Differentiable Fuzzy Neural Network for Recommender Systems
This project implements differentiable fuzzy neural networks for recommendation tasks.

## Requirements
- Python 3.10
- Required Python packages listed in `requirements.txt`
- Second environment with `requirements-ncf.txt` for `ncf-baseline.ipynb`

## Usage
1. Download the datasets by calling ```python download_data.py```
2. Install the necessary python packages ```pip install -r requirements.txt```
3. Train and evaluate a dataset by calling ```python main.py -c <config-file-path>```
4. As the surprise library requires numpy 1.x and recommenders require numpy 2.x, create a second environment for the NCF baseline
5. Install the necessary packages in the new environment ```pip install -r requirements-ncf.txt```
6. Run the Jupyter Notebook `ncf-baseline.ipynb`
