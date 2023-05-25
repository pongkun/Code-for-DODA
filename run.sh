python main.py --dataset cifar100 --imb_ratio 100 --num_max 500 --epochs 400 \
               --batch-size 256 --gpu 0 \
               --aug_prob 0.5 --loss_fn bs --doda --cutout --aug_type autoaug_cifar
