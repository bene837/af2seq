# AF2Seq: Alphafold backbone design pipeline
## Installation

Create a conda enviroment:  
`conda env create -f environment_cpu.yml`  

pyrosetta has to be installed seperately:  
`conda install -c https://NAME:PASSWORD@conda.graylab.jhu.edu pyrosetta`  
Please refer to the pyrosetta webpage for detailed instructions

Alphafold weigths can be dowloaded according to the instructions on the official AlphaFold repo.

Go to the repository folder and run:  
`pip install .`
#### GPU Version:
Install the cpu env first then add:  
`pip install --upgrade pip`  
`pip install "jax[cuda]>=0.2,<0.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`  

The code was tested using:    
`gcc                        8.4.0-cuda`  
`cuda                       11.1.1`  
`cudnn                      8.0.5.39-11.1-linux-x64`  

#### Starting sequence from secondary structure:
Please install dssp to enable the automatic starting sequence generation.  
`pip install pydssp `

Then call the following function
```python
from af2seq.design.utils import generate_start_sequence

sequence = generate_start_sequence('path/to/pdb/file')
```




## Usage

```python
from af2seq import GradientDesign,MCMCDesign

design = GradientDesign('path/to/weights','output/path')

design.design('path/to/pdb/file',
         iterations=500,
         lr=1e-3,)

mcmc = MCMCDesign('path/to/weights','output/path',random_seed=0,           
         mcmc_muts=1)

mcmc.design('path/to/pdb/file',
         iterations=500)
```
### Plotting
```python
from af2seq import plotting

plotting.plot_pred(design)
```

For bigger structures, requiring more memory, set the following variabels:  
`TF_FORCE_UNIFIED_MEMORY=1`  
`XLA_PYTHON_CLIENT_MEM_FRACTION=2.0`


### Jupyter Notebook

`jupyter notebook design.ipynb`

### Command Line
    usage: Af2Seq [-h] [-n NAME] [-m MODEL] [-c CHAINS [CHAINS ...]] [-it ITER] [-s SEED] [--lr LR]
                  [-r RECYCLES] [-cl CLAMP] [-am AA_MASK [AA_MASK ...]] [-fp FIX_POS [FIX_POS ...]]
                  [-dlp DISABLE_LOSS_POS [DISABLE_LOSS_POS ...]] [-esl ENABLE_SC_LOSS [ENABLE_SC_LOSS ...]]
                  [-st STARTSEQ [STARTSEQ ...]] [--msas MSAS [MSAS ...]] [-mm MCMC_MUTS] [-so SURF_OPTIM]
                  [-l LOSS [LOSS ...]] [-lw LOSS_WEIGHTS [LOSS_WEIGHTS ...]]
                  datadir target mode out
    
    Fixed backbone design using AlphaFold
    
    positional arguments:
      datadir               path to the directory that contains the Alphafold weights
      target                target pdb file that is used as groundtruth
      mode                  Gradient descent (gd) or MCMC (mcmc)
      out                   path to output directory
    
     optional arguments:
      -h, --help            show this help message and exit
      -n NAME, --name NAME  Name of the experiment
      -m MODEL, --model MODEL
                            Select a specifiy model. ptm or multimer
      -c CHAINS [CHAINS ...], --chains CHAINS [CHAINS ...]
                            chains that are targeted for design.
      -it ITER, --iter ITER
                            How many design steps should be performed
      -s SEED, --seed SEED  seed for mcmc
      --lr LR, --learning_rate LR
                            learning rate
      -l LOSS [LOSS ...], --loss LOSS [LOSS ...]
                            loss function that is used for the optimization process
      -lw LOSS_WEIGHTS [LOSS_WEIGHTS ...], --loss_weights LOSS_WEIGHTS [LOSS_WEIGHTS ...]
                            specifies the impact of each loss term
      -r RECYCLES, --recycles RECYCLES
                            AF recycles
      -cl CLAMP, --clamp CLAMP
                            FAPE loss clamp clips the loss of the distance between two residues is greater
                            than 10A
      -am AA_MASK [AA_MASK ...], --aa_mask AA_MASK [AA_MASK ...]
                            which amino acids to mask
      -fp FIX_POS [FIX_POS ...], --fix_pos FIX_POS [FIX_POS ...]
                            which indexes to mask
      -dlp DISABLE_LOSS_POS [DISABLE_LOSS_POS ...], --disable_loss_pos DISABLE_LOSS_POS [DISABLE_LOSS_POS ...]
                            disable backbone FAPE for these positions
      -esl ENABLE_SC_LOSS [ENABLE_SC_LOSS ...], --enable_sc_loss ENABLE_SC_LOSS [ENABLE_SC_LOSS ...]
                            which positions we want use sidechain FAPE in the loss
      -st STARTSEQ [STARTSEQ ...], --startseq STARTSEQ [STARTSEQ ...]
                            startseq. A for helix,V for b-sheet G for unordered
      --msas MSAS [MSAS ...]
                            MSA input path, None for no MSA
      -mm MCMC_MUTS, --mcmc_muts MCMC_MUTS
                            number of mutations introduced each MCMC round
      -so SURF_OPTIM, --surf_optim SURF_OPTIM
                            dont allow hydrophobic mutations on the surface
