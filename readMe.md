# Machine learning fun

This is my own implementations of a few interesting machine learning models from scratch in Python.


Inspiration: [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course/linear-regression).

# Overview

Implemented models / algorithms:
- Linear regression
- Logistic regression
- Sequential neural network (and several layers like one-channel 2d convolution, linear, max-pooling etc.)
- Genetic algorithm (currently solving only OneMax problem)
- Decision tree (accepts DataFrames with only bools)

Jupyter notebooks are simple demos of these models. Mostly these models are full of math coded using Numpy. I tested their performance against some popular libraries like `scikit-learn` and `TensorFlow`. Datasets are available from hyperlinks in the sources.

# Quickstart

I'm using Python 3.11 because TensorFlow doesn't support newer Python versions at the moment.

1. Download the train/test datasets from sources
2. Clone the repository
3. Create directory called `data`
4. Unzip and put datasets in `data` 
5. Explore .ipynb files in your IDE

# Sources

Train/test datasets:
- [Car Prediction data source](https://www.kaggle.com/datasets/amjadzhour/car-price-prediction)
- [Rain in Australia data source](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/data)
- [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)


Really usefull materials for learning linear models:
- [Mathematics of Gradient Descent 1](https://www.youtube.com/watch?v=jc2IthslyzM)
- [Mathematics of Gradient Descent 2](https://www.youtube.com/watch?v=sDv4f4s2SB8)
- [Stochastic Gradient Descent](https://www.youtube.com/watch?v=vMh0zPT0tLI)
- [Logistic Regression using Scikit-Learn](https://www.youtube.com/watch?v=aL21Y-u0SRs)


Must-learn Python libraries:
- [Numpy](https://numpy.org/doc/stable/index.html)
- [Matplotlib](https://matplotlib.org/)
- [Scikit-learn](https://scikit-learn.org/stable/index.html)
- [Pandas](https://pandas.pydata.org/)
