# Base Classes

`compile(inputset)` treats each item as one integer sample. Use `simulate` or
`predict` for one sample and `simulate_many` or `predict_many` for a sequence.

```python
from concrete_fhe_toolkit.ml import FHELinearRegression

model = FHELinearRegression(weights=[2, 1], bias=1)
model.compile([[0, 0], [5, 5]])
assert int(model.simulate([3, 2])) == 9

# Build a fixed-size batch circuit from individual calibration samples.
model.compile([[0, 0], [5, 5]], batch_size=4, inputset_is_batched=False)
predictions = model.simulate_many([[1, 2], [3, 4], [5, 5]])
```

For existing prebatched inputsets, `compile(batches, batch_size=4)` retains its
meaning. Alternatively, `compile(batches, inputset_is_batched=True)` infers the
batch size. Partial prediction batches repeat their final sample for padding and
discard padded outputs. Calibration samples and runtime samples must have matching
shapes and cover the values used during inference.

::: concrete_fhe_toolkit.ml.classes
