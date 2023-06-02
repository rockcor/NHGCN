#!/bin/sh
# unzip data.zip
# pip install -r requirement.yaml
# run Reproduce.sh
python main.py --dataset cora --lr 0.1 --wd 0.001 --dropout 0.8 --finaldp 0 --hops 1 --includeself 0 --addself 1 --threshold 2
python main.py --dataset citeseer --lr 0.001 --wd 0 --dropout 0.8 --finaldp 0.9 --hops 2 --includeself 1 --addself 1 --threshold 2
python main.py --dataset pubmed --lr 0.1 --wd 0.0001 --dropout 0.8 --finaldp 0.6 --hops 1 --includeself 0 --addself 1 --threshold 2.5
python main.py --dataset photo --lr 0.1 --wd 0.0001 --dropout 0.7 --finaldp 0.5 --hops 1 --includeself 1 --addself 1 --threshold 3.5
python main.py --dataset computers --lr 0.05 --wd 0 --dropout 0 --finaldp 0 --hops 1 --includeself 0 --addself 0 --threshold 4
python main.py --dataset film --lr 0.1 --wd 0.001 --dropout 0.8 --finaldp 0.5 --hops 2 --includeself 0 --addself 1 --threshold 3
python main.py --dataset chameleon_filtered --lr 0.1 --wd 0.001 --dropout 0.6 --finaldp 0 --hops 2 --includeself 0 --addself 0 --threshold 3
python main.py --dataset squirrel_filtered --lr 0.1 --wd 0.001 --dropout 0.7 --finaldp 0.6 --hops 1 --includeself 1 --addself 0 --threshold 3.5
python main.py --dataset texas --lr 0.1 --wd 0.005 --dropout 0.7 --finaldp 0.6 --hops 2 --includeself 0 --addself 1 --threshold 3.5
python main.py --dataset cornell --lr 0.1 --wd 0.001 --dropout 0.7 --finaldp 0.6 --hops 1 --includeself 0 --addself 1 --threshold 3.5
