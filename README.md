# PLH in GL
This project implements Persistent Local Homology in Graph Learning. The code draws inspiration from the following two projects:  
https://github.com/pkuyzy/TLC-GNN  
https://github.com/snap-stanford/ogb

## Requirements
Main packages required are as follows.
```
Python=3.7.13
pytorch=1.12.1
cuda=11.3
cudnn=8
```
Other dependencies listed in the Dockerfile.

## Experiments

### OGBN-ARXIV
```commandline
python ogbn_arxiv_gnn.py conf.d/arxiv_template.yaml
```

### OGBL-DDI
```commandline
python ogbl_ddi_gnn.py conf.d/ddi_template.yaml
```

### PPI
```commandline
python plh_gnn_lp.py conf.d/ppi_template.yaml
```

## License
OGB: MIT  
TLC-GNN: N/A  
This Project: MIT