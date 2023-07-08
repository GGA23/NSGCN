# UGCN-SN
A PyTorch implementation of UGCN-SN "Universal Graph Convolutional Networks with Selective Neighbors". <br>
code is coming soon
# Environment Settings
This implementation is based on Python3. To run the code, you need the following dependencies: <br>
* torch==1.8.1
* torch-geometric==1.7.2
* scipy==1.2.1
* numpy==1.19.5
* tqdm==4.59.0
* seaborn==0.11.2
* scikit-learn==0.24.2
* CUDA Version: 11.0
# Datasets
The data folder contains five homophilic benchmark datasets(Cora, Citeseer, Pubmed, Computers, Photo), and five heterophilic datasets(Chameleon, Squirrel, Cornell, Texas, Wisconsin) from [BernNet](https://github.com/ivam-he/BernNet) and [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn). We use the same experimental setting (60\%/20\%/20\% random splits for train/validation/test with the same random seeds, epochs, run ten times, early stopping) as [BernNet](https://github.com/ivam-he/BernNet).   
# Run an experiment:
    $ python train.py --dataset Chameleon
    $ sh demo.sh
# Examples
 Training a model on the default dataset.  
<iframe height=498 width=510 src="https://github.com/GGA23/UGCN-SN/blob/main/demo.mp4">

# Baselines links
* [H2GCN](https://github.com/GitEventhandler/H2GCN-PyTorch)
* [MixHop](https://github.com/benedekrozemberczki/MixHop-and-N-GCN)
* [GCNII](https://github.com/chennnM/GCNII)
* [GPRGNN](https://github.com/jianhao2016/GPRGNN)
* [BernNet](https://github.com/ivam-he/BernNet)
* [GloGNN](https://github.com/RecklessRonan/GloGNN)
* The implementations of others are taken from the Pytorch Geometric library
# Acknowledgements
The code is implemented based on [BernNet: Learning Arbitrary Graph Spectral Filters
via Bernstein Approximation](https://github.com/ivam-he/BernNet).
