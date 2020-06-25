# TpTnOsc
A Python package for analyzing totally positive (TP), totally non-negative (TN), and oscillatory (OSC) matrices.

The tool contains many functions and classes that can perform the following:
* compute matrix minors and multiplicative compound matrices
* compute and display the SEB factorization of an Invertible TN (I-TN) matrix
* test if a matrix is TP, TN, I-TN, or OSC
* compute the number of (cyclic and non-cyclic) sign variations in a vector
* compute the exponent of an oscillatory matrix
* compute and display families of vertex-disjoint paths corresponding to lower-left and upper-right corner minors of oscillatory matrices

See [notebook](https://github.com/yoramzarai/TpTnOsc/blob/master/examples/osc_exp_examples.ipynb) for details and examples.

## Modules
- utils.py contains several functions related to the number of sign variations in a vector, multiplicative compound matrices, SEB factorization, planar networks and other TP/TN/OSC matrices related functions.
- osc_exp.py contains the functions and classes to compute the exponent of an oscillatory matrix and the corresponding families of vertex-disjoint paths.


## Examples
- osc_exp_examples.ipynb contains examples of using the triangle graph (derived from the planar networks of an SEB factorization) to deduce the exponent of an oscillatory matrix ![A](https://render.githubusercontent.com/render/math?math=A), and to compute the families of vertex-disjoint paths of each corner minor of ![A^r](https://render.githubusercontent.com/render/math?math=A%5Er), where ![r](https://render.githubusercontent.com/render/math?math=r) is the exponent of ![A](https://render.githubusercontent.com/render/math?math=A)

- basic_examples.py contains few basic examples

- mat_call_py_example.m is a Matlab example of using Python functions from utils.py

## Install
The simplest way to install TpTnOsc is through pip or conda:

```
pip install TpTnOsc

# or

conda install --channel "yoramzarai" TpTnOsc
```

Alternately, TpTnOsc can be installed from github:
```
git clone https://github.com/yoramzarai/TpTnOsc.git

cd TpTnOsc

python setup.py install
```

