#CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 lib/train_val.py &> monouni_train_eval.log&
python lib/train_val.py