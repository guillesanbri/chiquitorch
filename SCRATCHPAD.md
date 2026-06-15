# Scratchpad

- `torch.fx.Graph` -> Ordered list of nodes describing dataflow. No weights, can't be run by itself. E.g. `ep.graph`.
- `torch.fx.GraphModule` -> `nn.Module` subclass, pairs a Graph with the parameters and buffer tensors.
- `torch.export.ExportedProgram` -> Bundle of `.graph_module` (in lifted form), `.graph` (shortcut for `.graph_module.graph`), `.graph_signature`, `state_dict`, etc.

> Lifted form means that every parameter and buffer of the module is turned into an explicit input placeholder at the front of the argument list. It's the difference between `ep.module()` and `ep.graph_module` (with the latter one the lifted version).

## Phases

1.- Desktop PC, Python (we are here)
2.- Desktop PC, C++, weight only quantization, basic optimizations
3.- Microcontroller basics (ESP32-S3/P4?), memory planning
4.- Microcontroller (look for devkits, see below, M55?), kernel optimizations
5.- M55 + U55 (?), heterogeneous compute?, graph partitioning

> Maybe multiple device parallelism on something cheaper like the ESP32 is more interesting as a first stretch instead of the heterogeneous compute. Either way phase 1/2 have plenty of room for work.

## Microcontrollers

- As flexible as possible
- Debug interface
- Toolchains spooky
- Study in which ones it makes more sense to do FP16/INT8
- Seems like ESP32 [supports it](https://docs.espressif.com/projects/esp-dl/en/release-v1.1/esp32s3/tools/quantization-toolkit/quantization-specification.html)
- Cortex-M55/M-85 look good, probs the M5 is enough, helium extension (?)
- NUCLEO-F031K6? -> Cortex-M0, also P-NUCLEO-WB55
- CY8CKIT-062S2-AI
- Teensy 4.1 -> Cortex-M7
- Grove - Vision AI Module V2 -> maybe not the most transparent flexible. Look into the U55 NPU to see if we can access it directly instead of through vendor code
- The alif boards look good if they become available: DK-E7/E8 (220€), SK-E1C (43€), DK-E1C (133€)
- https://hc32.hotchips.org/assets/program/conference/day1/HotChips2020_Edge_Computing_Arm_Cortex-M55.pdf

## References to look at:

- TVM
- TensorRT
- [LiteRT](https://developers.google.com/edge/litert), [Model optimization](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/model_optimization), [for Microcontrollers](https://developers.google.com/edge/litert/microcontrollers/overview)
- ExecuTorch
- Glow
- IREE
- NVIDIA Model-Opt
- XNNPACK
