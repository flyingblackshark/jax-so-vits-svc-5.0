English | [中文教程](https://github.com/flyingblackshark/jax-so-vits-svc-5.0/blob/Main-5/README_zh_cn.md)
# SO-VITS-SVC 5.0 IN JAX
The following tutorials are for Google TPU v2-8/v3-8
# Acknowledgements
This project is supported by Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).
## Prepare Environment
	pip install -r requirements.txt
	pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
	sudo apt update && sudo apt install -y libsndfile1 ffmpeg
## Prepare Dataset
Dwonload pretrained models from [so-vits-svc-5.0](https://github.com/PlayVoice/so-vits-svc-5.0)
Generate or copy your data_svc folder from [so-vits-svc-5.0](https://github.com/PlayVoice/so-vits-svc-5.0)
Replace HuBERT with [latest model](https://github.com/bshall/hubert/releases/download/v0.2/hubert-soft-35d9f29f.pt)
## Train Your Model
	python3 prepare/preprocess_train.py
	python3 svc_trainer.py
## Inference
	python3 svc_inference.py --config configs/base.yaml --spk xxx.spk.npy --wave test.wav
## Join us
	[Discord Channel](https://discord.gg/mrGUhMVWUM)