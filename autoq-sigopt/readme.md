# autoq-sigopt
* The codes in this folder aims to demonstrate SigOpt's Bring Your Own Optimizer (BYOO) capability with AutoQ. 
* AutoQ backend has been integrated with the API calls to SigOpt.
* Imagenet AutoQ mixed precision search has been adapted to allow user to input thier credentials for logging and visualization.

# Install
```
git clone https://github.com/vuiseng9/nncf
cd nncf && git checkout autoq-sigopt
python setup.py develop
pip install -r examples/torch/requirements.txt
pip install -r autoq-sigopt/requirements.txt
```

# Run
```
cd nncf/autoq-sigopt/
# modify, review and run run_autoq_sigopt_imagenet.sh
```
You can check your page in SigOpt.com