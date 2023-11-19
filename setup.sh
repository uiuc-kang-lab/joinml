mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
. ~/.bashrc
conda create -n joinml
conda activate joinml
sudo apt-get update
sudo apt-get install -y libgl1
conda install -y python=3.10.12
pip install -r requirements.txt
pip install -e .
