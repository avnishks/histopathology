## Run the WSI preprocessing pipeline notebook (inside docker):

- The script assumes the following folder structure:

```bash
.
├── /datasets
│    └── Camelyon17
│         └── training
|              ├── center_0
|              ├── center_1
|              └── center_2
│   
└── /notebooks
     └── WSI preprocessing pipeline.ipynb
```

- If your folder structure is different, please update the `BASE_DIR` and `sample_slide` variables in cell #2 and cell #4, respectively, before proceeding with the notebook.