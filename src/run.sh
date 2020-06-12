#!/bin/bash

# exec python3 main_fonduer_electricity.py --docs 114 --exp "full" --clear_db 1 --cls_methods "rule-based, logistic-regression, lstm" >> results/elec_norm_lstm.txt &
# exec python3 main_fonduer_electricity.py --docs 114 --exp "full_pred" --clear_db 1 --cls_methods "rule-based, logistic-regression, lstm" >> results/elec_norm_lstm.txt &
exec python3 main_fonduer_electricity.py --docs 114 --exp "gold_pred" --clear_db 1 --cls_methods "rule-based, logistic-regression, lstm" >> results/elec_pred_lstm.txt &
exec python3 main_fonduer_electricity.py --docs 114 --exp "gold" --clear_db 1 --cls_methods "rule-based, logistic-regression, lstm" >> results/elec_gold_lstm.txt