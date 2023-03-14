# MTLCosPrune
We investigate filter pruning in the multi-task neural networks using a proposed filter pruning method called **CosPrune** that is specfically designed for the multi-task models and compare it with another popular single-task filter pruning method called [Taylor Pruning](https://github.com/NVlabs/Taylor_pruning). In the iterative pruning setting our proposed method perform better than the baseline Taylor pruning method. 

<p float="left">
  <img src="https://github.com/gargsid/MTLCosPrune/blob/main/assets/iter.png" width="1200" height="300" />
</p> 

We also trained the pruned models from scratch after random initialization for both the methods only to observe that the model's performance does not depend on the architecture design but on the initialization and the appropriate hyperparameter settings. 

<p float="left">
  <img src="https://github.com/gargsid/MTLCosPrune/blob/main/assets/one-shot.png" width="1200" height="300" />
</p> 

## Running the code
- Training on NYUv2 dataset: `train_base_model.py`
- Pruning code: `main.py`
- Training pruned models: `train_config_models.py`

## Acknowledgments
This code uses the Taylor Pruining implemenation from the [original repository](https://github.com/NVlabs/Taylor_pruning)
