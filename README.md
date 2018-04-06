# Kernel Embedding-based Nonstationary Causal Model Inference

Python code of causal inferene algorithm for **cause-effect pairs** proposed in paper *A Kernel Embedding–Based Approach for Nonstationary Causal Model Inference* (ENCI).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
- NumPy
- SciPy
- scikit-learn

We test the code using **Anaconda3 5.0.1 64-bit for python 3.6.3** on windows. Any later version should still work perfectly. The download page of Anaconda is [here](https://www.anaconda.com/download/).

## Running the tests

After installing all required packages, you can run *test.py* to see whether **ENCI** could work normally.

The test code does the following:
1. it generate 100 groups of synthetic data and put them in a list. 
(Each group is an L-by-2 *numpy array* where L is the number of points ranging from 40 to 49. The first column of each array is the cause and the other column being the effect.)
2. ENCI is applied on the generated data set to infer the causal direction.


## Apply **ENCI** on your data

### Usage

Import **ENCI** using

```
from ENCI import cd_enci
```

Apply **ENCI** on your data

```
direction = cd_enci(XY)
```

### Notes

Input of function **cd_enci()**

| Argument  | Description  |
|---|---|
|XY | List of numpy arrays with 2 columns and any number of rows. Rows of each array represent i.i.d. samples, each column corresponds to a variable in the cause-effect pair.|

Output of function **cd_enci()**

| Argument  | Description  |
|---|---|
|direction | 1 - the first column is the cause; -1 - the second column is the cause|

## Authors

* **Shoubo Hu** - shoubo.sub@gmail.com
* **Zhitang Chen** - chenzhitang2@huawei.com

See also the list of [contributors](https://github.com/amber0309/ENCI_cause-effect-pair/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
