# fastmtl
> Multi-task learning utilities for fastai


## Install

`pip install fastmtl`

## Usage

### Loss

Apply a loss function per model output and get weighted sum of them. For instance, if the first model output is for classification and the second model output is for regression,
```py
from fastmtl.loss import CombinedLoss
loss_func = CombinedLoss(CrossEntropyLossFlat(), MSELossFlat(), weight=[1.0, 3.0])
```

### Metric

Apply metrics for each model output. For instance, if we have a model making classification and regression, we can evaluate each model output with relevant metrics. Assuming that model outputs a tuple of tensors for classification and regression, respectively:

```py
from fastai.metrics import F1Score, R2Score
from fastmtl.metric import mtl_metrics

clf_f1_macro =  F1Score(average='macro')
clf_f1_macro.name = 'clf_f1(macro)'
clf_f1_micro =  F1Score(average='micro')
clf_f1_micro.name = 'clf_f1(micro)'

reg_r2 = R2Score()
reg_r2.name = 'reg_r2'

# metrics for classification in the first list 
# metrics for regression in the second list 
metrics = mtl_metrics([clf_f1_macro, clf_f1_micro], [reg_r2])

learn = Learner(
    ...
    metrics=metrics,
)
```

## Tutorials

[Video distortion detection](https://bdsaglam.github.io/fastmtl/tutorial.vqa)

## TODO
- [ ] Support tabular learner
- [ ] Support fastai>=2.7
