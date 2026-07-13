# Conda installation for SOFI compatibilty
As of 13/07/2026, the `conda_requirements.txt` file contained in this directory is useful for installing a conda environment that is compatible for using SOFI. 
[SOFI](https://mammasmias.github.io/IterativeRotationsAssignments/index.html) is an algorithm to extract molecule symmetry operations automatically, analogous to `spglib` for extracting substrate symmetry.
The conda environment can be installed using the following command:
```
conda create --name <env> --file conda_requirements.txt
```
