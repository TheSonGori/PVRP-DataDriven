python -u scripts/vns_multi_seed.py --instancias p01 p03 p04 p06 p09 --iters 8000 | Tee-Object -FilePath results/final_8000_grupo1.txt
python -u scripts/vns_multi_seed.py --instancias p02 p07 --iters 8000 | Tee-Object -FilePath results/final_8000_grupo2.txt
python -u scripts/vns_multi_seed.py --instancias p05 p08 --iters 8000 | Tee-Object -FilePath results/final_8000_grupo3.txt