

```bash
git clone https://github.com/vuiseng9/nncf
cd nncf
git checkout v2.5.0-ptq-rn50
pip install .[torch]
cd examples/post_training_quantization/torch/resnet50
pip install -r requirements.txt

# do revise imagenet dataset_path in the script
python main_rn50_imgnet.py

# this is based on mobilenet_v2/main.py - diff to see changes
```