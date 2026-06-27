# !/bin/bash

# python3 main.py --model_name rnn  --dataset_name mnist --hidden_size 512 --input_size 1 --output_size 10 --nepochs 50 --nruns 5 --weight_decay 0.000 --lr 1e-5  --exponential_decay True --continue_training False 

# python3 main.py --model_name gru  --dataset_name mnist --hidden_size 512 --input_size 1 --output_size 10 --nepochs 50 --nruns 5 --weight_decay 0.000 --lr 1e-5  --exponential_decay True --continue_training False 

# python3 main.py --model_name lstm  --dataset_name mnist --hidden_size 512 --input_size 1 --output_size 10 --nepochs 50 --nruns 5 --weight_decay 0.000 --lr 1e-5  --exponential_decay True --continue_training True 

# python3 main.py --model_name urlstm  --dataset_name mnist --hidden_size 512 --input_size 1 --output_size 10 --nepochs 50 --nruns 5 --weight_decay 0.000 --lr 1e-3 --continue_training False 

# python3 main.py --model_name hippo  --dataset_name mnist --hidden_size 512 --input_size 1 --output_size 10 --nepochs 50 --nruns 5 --weight_decay 0.000 --lr 1e-3 --continue_training False 

python3 main.py --model_name gatedhippo --dataset_name mnist --hidden_size 512 --input_size 1 --output_size 10 --nepochs 50  --nruns 5 --weight_decay 0.00 --lr 1e-3 --continue_training False
