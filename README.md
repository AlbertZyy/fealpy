# FEALPy:Finite Element Analysis Library in Python 
![Python package](https://github.com/weihuayi/fealpy/workflows/Python%20package/badge.svg)
![Upload Python Package](https://github.com/weihuayi/fealpy/workflows/Upload%20Python%20Package/badge.svg)

![](./FEALPY.png)

While beginning with the finite element algorithm, FEALPy's sights are set on
exploring vast horizons.

We hope FEALPy will be an open-source library for intelligent CAX 
algorithms, integrating CAX fundamentals with AI to support advanced algorithm
research and the cultivation of versatile talent.

We also hope FEALPy can accelerate the creation and testing of next-gen
intelligent CAX apps, paving the way for advanced algorithms in industrial
applications.

So FEALPy's development goal is to become the next generation intelligent CAX
computing engine.

The word "FEAL" is an archaic or poetic term in English, meaning faithful or
loyal. Though not commonly used in modern English, it carries strong
connotations of unwavering dedication and reliability.

The name "FEALPy" embodies this essence of loyalty and faithfulness. It
signifies the software's commitment to being a dependable and trustworthy tool
in the field of intelligent CAX computation. Just as "FEAL" suggests
steadfastness, FEALPy aims to provide consistent, reliable support for
researchers, engineers, and developers in their pursuit of innovative solutions
and advancements in CAX computation. The name reflects the software's mission to
be a loyal companion in the journey toward groundbreaking discoveries and
industrial applications.

# Installation

## Miniconda

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

```bash
conda create -n gpufealpy310 python=3.10
conda activate gpufealpy310
conda install numpy=2.0.1 -c conda-forge #2.0.1
conda install ipython notebook -c conda-forge
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia # 0.4.31
conda install cupy -c conda-forge  -c nvidia
conda install pytorch=2.3.1 -c conda-forge -c nvidia

```

## From Source (Recommanded)

First, clone the FEALPy repository from GitHub

```bash
git clone https://github.com/weihuayi/fealpy.git
```

If you can't access GitHub, you can clone it from Gitee
```bash
git clone https://gitee.com/whymath/fealpy
```

It is recommended to create a virtual environment to manage dependencies:
```bash
python -m venv fealpy_env
source fealpy_env/bin/activate  # On Windows, use `fealpy_env\Scripts\activate`
```

Then change directory to the cloned repository and install FEALPy in editable(`-e`) mode:
```bash
cd fealpy
pip install -e .
```

If you want to install optional dependencies, such as `pypardiso`, `pyamg`,
`meshpy` and so on, you can do so by specifying the [optional] extra:
```
pip install -e ".[optional]"
```

To install both development and optional dependencies, use:
```bash
pip install -e ".[dev,optional]"
```
To verify that FEALPy is installed correctly, you can run the following command:

```bash
python -c "import fealpy; print(fealpy.__version__)"
```

To update your FEALPy installation to the latest version from the source repository, navigate to the FEALPy directory and pull the latest changes:
```bash
cd fealpy
git pull origin main
```

To uninstall FEALPy, just run the following command:
```bash
pip uninstall fealpy
```

## Development
For FEALPy developers, the first step is to create a **fork** of the https://github.com/weihuayi/fealpy repository in your own Github account. 

Clone the FEALPy repository under your own account to the local repository: 
```bash
# replace<user name>with your own GitHub username
git clone git@github.com:<user name>/fealpy.git 
```

> Note that the following operations need to be operated in the fealpy folder.

Set up the upstream repository: 
```bash
git remote add upstream git@github.com:weihuayi/fealpy.git
```

Before local development, need to pull the latest version from the upstream repository and merge it into the local repository:  
```bash
git fetch upstream
git merge upstream/master
```

After local development, push the modifications to your own remote repository:
```bash
git add modified_files_name
git commit -m "Explanation on modifications"
git push
```

Finally, in your own Github remote repository, open a **pull request** to the upstream repository and wait for the modifications to be merged. 

## Warning 
The sparse pattern of the matrix `A` generated by `FEALPy` may not be the same as the theoretical pattern, since there exists nonzero values that are close to machine precision due to rounding. If you care about the sparse pattern of the matrix, you can use the following commands to eliminate them
```python
eps = 10**(-15)
A.data[ np.abs(A.data) < eps ] = 0
A.eliminate_zeros()
```

## Docker

To be added.

## Reference and Acknowledgement

We thank Dr. Long Chen for the guidance and compiling a systematic documentation for programming finite element methods.
* http://www.math.uci.edu/~chenlong/programming.html
* https://github.com/lyc102/ifem


## Citation

Please cite `fealpy` if you use it in your paper

H. Wei and Y. Huang, FEALPy: Finite Element Analysis Library in Python, https://github.com/weihuayi/fealpy, *Xiangtan University*, 2017-2024.

```bibtex
@misc{fealpy,
	title = {FEALPy: Finite Element Analysis Library in Python. https://github.com/weihuayi/fealpy},
	url = {https://github.com/weihuayi/fealpy},
	author = {Wei, Huayi and Huang, Yunqing},
    institution = {Xiangtan University},
	year = {Xiangtan University, 2017-2024},
}
```









