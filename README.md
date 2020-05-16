# bin2sth

Under coding...

``` python
import sys; sys.path.append(['/Users/ligengwang/Projects/pycharm/bin2sth']); import logging; logging.getLogger().setLevel('WARN')
```

## Hashable Args

### How to identify a dataset, model and metrics?

- Dataset
  - vocab: `BinArg`
  - corpus: `BinArg`
  - setting:
    - i.e. window, sub-sampling
- Model
  - Dataset
  - Model Setting
    - i.e. n_emb, n_hdn
  - Runtime Setting
    - i.e. epoch, lr
- Result
  - Model
  - Metrics

### Args Category

There are several kinds of args. By classifying them, we can take these
args as the key to store or load corresponding data, model and results.

- Dataset Args
  - used to identify data
  - components
    - vocab: `BinArg`, corpus: `BinArg`
    - preprocessing: a chain of `Pp` 
- Train Args
  - used to identify model
  - components
    - Dataset Args: specify on which data source the model is trained
    - Model Args: args that influence the model construction, such as 
    the network size
    - Runtime Args: args that control the training process
- Result Args 
  - Model Args
