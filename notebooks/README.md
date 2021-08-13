# Notebooks

This folder regroups all notebooks related to the project. They can be either for visualisation, evaluation, or training.

## Automatic generation of jupyter notebooks
In the optic of research reproducibility and easy science, we can generate  and execute `.ipnb` on the fly. 

Steps to follow:
1. Make sure you are using `mlflow`
2. `pip install nbformat nbconvert jupyter_client ipykernel`
3. In `notebooks/auto_ipnb.py`, add your own notebook template.
    
   All the notebooks-like function should:
   - be self contained.
   - Markdown cells should be inside  \"\"\"...\"\"\" 
   - Code cells should be inside `# <code> ... # </code>`
   - The template variables should be defined as `{{cookie}}`

   Please refer to the `notebooks/auto_ipynb.saliency` function for an example.
4. Add `mlflowipynb` to your list of callbacks. For example:
    ```
    callbacks:
        - mlflowHistory: {}
        - mlflowModelCheckPoint: {}
        - mlflowipynb:
            execute: True
            func: saliency
            cookies:
              dataset_cfg:
                dataset: *dataset_cfg
              num_batch: 1
              layer_name: 'dense_1'
    ```
    This will generate a `saliency.ipynb` and a `saliency.html` files.
    Theses file are then visible as `mlflow` artifacts, directly in the experiment page.*
    
    \* `mlflow` doesn't support `html` viewer (for now). While waiting for it's integration, you can either:
        
    - Learn `node.js` and add it yourself (:D ).
    - Ask `francis.dutil@imagia.com` how to do it.
    
### Saliency

Saliency notebook. Load the model, and extract the saliency map for different examples for a given dataset.

#### Magic variables:
- `run_uuid`: Experiment unique Id. Normally it's passed automatically. 
- `dataset_cfg`: The dataset configuration. Can either copy the `dataset` section in the yaml file, or (and recommand) use `yaml` anchor.
- `num_batch`: The number of minibatch to evaluate.
- `layer_name`: The layer from which to compute the saliency map. ex: `dense_1`.

#### Dependencies:
- `pip install matplotlib`
- `pip install git+https://gitlab.com/imagia-research/dark-knight/weakloc/keras-vis.git`