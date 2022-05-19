bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name baseline_cnn
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name baseline_cnn --normalize
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name baseline_cnn --normalize --horizontal_flip
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name baseline_cnn --normalize --vertical_flip
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name baseline_cnn --normalize --rotate
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name baseline_cnn --normalize --perspective
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name baseline_cnn --normalize --color_jitter
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name baseline_cnn --normalize --random_crop
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name baseline_cnn --normalize --horizontal_flip --vertical_flip --rotate --color_jitter --random_crop --perspective

bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name resnet18_cnn
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name resnet18_cnn --normalize
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name resnet18_cnn --normalize --horizontal_flip
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name resnet18_cnn --normalize --vertical_flip
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name resnet18_cnn --normalize --rotate
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name resnet18_cnn --normalize --perspective
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name resnet18_cnn --normalize --color_jitter
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name resnet18_cnn --normalize --random_crop
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name resnet18_cnn --normalize --horizontal_flip --vertical_flip --rotate --color_jitter --random_crop --perspective

bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name resnext50_cnn
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name resnext50_cnn --normalize
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name resnext50_cnn --normalize --horizontal_flip
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name resnext50_cnn --normalize --vertical_flip
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name resnext50_cnn --normalize --rotate
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name resnext50_cnn --normalize --perspective
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name resnext50_cnn --normalize --color_jitter
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name resnext50_cnn --normalize --random_crop
bsub -n 4 -W 24:00 -R "rusage[mem=16384]" python task2.py --model_name resnext50_cnn --normalize --horizontal_flip --vertical_flip --rotate --color_jitter --random_crop --perspective