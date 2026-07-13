# Conda installation for SOFI compatibilty
As of 13/07/2026, the following steps are recommended for installing an environment that is compatible with [SOFI](https://mammasmias.github.io/IterativeRotationsAssignments/index.html).
SOFI is an algorithm to extract molecule symmetry operations automatically, analogous to `spglib` for extracting substrate symmetry.
1. Use `conda` to create an environment with `python=3.11` and `pip`.
2. Activate the environment.
3. Install SOFI using conda with `conda install ira -c conda-forge`.
4. Install BOSS using `pip`.
4. Install the required packages using `requirements.txt`
5. Navigate to the root directory of the symmetry analysis algorithm.
6. Install the algorithm using `pip install .`
