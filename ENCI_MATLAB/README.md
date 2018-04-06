# Kernel Embedding-based Nonstationary Causal Model Inference

MATLAB code of causal discovery algorithm for **cause-effect pairs** proposed in paper [A Kernel Embedding–Based Approach for Nonstationary Causal Model Inference](https://www.mitpressjournals.org/doi/abs/10.1162/neco_a_01064) (ENCI).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

We test the code using **MATLAB R2017b** on windows. Any later version should still work perfectly.

## Running the tests

In MATLAB, change your *current folder* to "ENCI_MATLAB/" and run the file *example.m* to see whether it could run normally.

The test does the following:
1. it generate 100 groups of a cause-effect pair and put all groups in a MATLAB *cell array*.
(Each group is a L by 2 matrix where L is the number of points ranging from 40 to 50.)
2. ENCI is applied on the generated data set to infer the causal direction.

## Apply on your data

### Usage

Change your current folder to "ENCI_MATLAB/" and use the following commands

```
order = ENCI_pairs(X)
```

### Description

Input of function **ENCI_pairs()**

| Argument  | Description  |
|---|---|
|X | Cell array of matrix. Rows of each matrix represent i.i.d. samples, each column corresponds to a variable in the cause-effect pair.|

Output of function **ENCI_pairs()**

| Argument  | Description  |
|---|---|
|order | The estimated causal direction.<br/>If denote the variable of the first column by X and the second by Y, then <br/>1:  X --> Y <br/>-1: Y --> X|

## Authors

* **Shoubo Hu** - shoubo.sub@gmail.com

See also the list of [contributors](https://github.com/amber0309/ENCI_graph/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

