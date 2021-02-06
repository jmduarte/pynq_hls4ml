# pynq_hls4ml

## Install and setup Vivado
Tested with Vivado 2019.1 and 2019.2.
```
source /<Xilinx installation directory>/Vivado/<version>/settings64.sh 
```

## Create conda environment
```
conda-env create -f environment.yml
```

## Launch jupyter notebook and run through 
1. `part1_training_to_bitfile.ipynb` (on the host CPU)
2. `part2_pynq.ipynb` (on the PYNQ)
3. `part3_compare.ipynb` (on the host CPU)
