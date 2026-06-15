# chiquitorch

A toy PyTorch export backend using codegen.

## Expected usage / API

```python
from chiquitorch import compile

compile(model, inputs, output_dir="generated/")
```

`compile` will:

```python
def compile(model, inputs, output_dir="generated/"):
    # export an FX graph
    exported = torch.export.export(model, inputs)
    
    # optimize and lower the graph
    ir = lower(exported.graph_module)
    ir = quantize(ir)
    ir = memory_plan(ir)
    
    # emit the generated code
    emit(ir, output_dir)
```

> The first stage is a full-python pipeline without any kind of graph/inference optimization, final goal is to move to microcontrollers.

## ToDo

- [ ] Basic codegen
- [ ] Basic pure-python backend
- [ ] Test suite to validate matching outputs
- [ ] Constant folding
- [ ] Dead code elimination
- [ ] In-place ops
- [ ] Memory planning
- [ ] Operation fusion
- [ ] Quantization

### Testing

- [ ] Write small model zoo: basic ops, mlp, conv block, residual connection, mha
- [ ] Write harness abstraction (`assert_close` for now) with HIL testing in mind. Decide between random inputs and pre-computed results.
- [ ] Write a dummy interpreter to validate the testing pipeline.

### Docs

- [ ] Explain codegen vs. interpreter/dispatcher

### At some point

- [ ] Multi-device parallelism
- [ ] Read about loop tiling, weight compression, subgraph scheduling, layout optimization
- [ ] Interpreter/dispatcher + inference engine version

## References

- [torch.fx](https://docs.pytorch.org/docs/2.12/fx.html)
- [torch.export](https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/export.html)