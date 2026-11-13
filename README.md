front end :

* hls4ml is capable of processing quantized models from all major deep learning frameworks (**PyTorch, ONNX, Keras**)

back end :

Currently available : \[1]

* Vitis and Vivado : **VITIS HLS**

  * deprecated counterpart (old use) Vivado HLS : **AMD/Xilinx FPGAs**
* oneAPI and Quartus : **oneAPI**

  * deprecated counterpart (old use) Intel HLS : **Intel/Altera FPGAs**
* Catapult : **Catapult HLS** targeting the Siemens EDA flow \[2]





pragma s control configs



* hls4ml converts the model to an **internal representation (IR)** before **optimizers** modify it to suit the hardware architecture
* For pytorch, since its a channels-first representation \[Batch, Height, Width, RGB], and hls4ml supports a channels-last representation (because it was made for Keras 2), there are optimizers that automatically inject transpose operations into its internal blueprint **(IR)**

  * Brevitas is a pytorch library used for training Quantized Neural Networks **(QNN)** or **QAT**

&#x09;1. You take your Brevitas-trained PyTorch model.

&#x09;2. You use Brevitas's built-in exporter to turn it into a QONNX file (a specialized version of the

&#x09;universal ONNX standard that includes specific blocks for quantization).

&#x09;3. The hls4ml ONNX front end reads this QONNX file, understands the low-bit math,

&#x09;and prepares it for hardware.

* hls4ml transforms its models to files that are uniform for its own system (universal) it all becomes one file, however, there is the object MultipleModelGraph (instead of ModelGraph) to be able to do parrallel independent synthesis (reducing synthesis time or to distribute large models over multiple chips/devices)
* ***ModelGraph*** object hold directives in ***HLSConfig*** that arent in the model itself (I/O type, strategy implementation, target back end)
* Precision in HLSConfig: Accumulator precision (holds intermediate MAC results) isn't usually in the front-end model, so since v1.0.0 you can set precision to "auto" (conservatively estimates bits to avoid overflow). In v1.2.0, an optimizer pass uses interval arithmetic to propagate exact bit-widths across fully quantized models (ignores user precision configs for bit-exactness).







**Hardware Implementations**

* Back ends generate HLS code using manually optimized C++ templates for each layer + generate utility files (TCL synthesis scripts, test benches, Python bridge code, and synthesis report parsers).
* Constant Matrix-Vector Multiplication (CMVM): core math block of neural nets. Parallelism is controlled by Reuse Factor (RF) -> for an MxN weight matrix, max parallel multiplications per clock cycle is N\_MULT = (M \* N) / RF. RF sets the Initiation Interval (II = how many clock cycles before the layer can accept new data).
* &#x20;**3 CMVM Strategies:**

&#x09;- Latency strategy: fully unrolls CMVM loops, weights are hardcoded directly into multiplier circuits. Compiler optimizes out zeros and turns constants into shift-and-add. Best for small models / ultra-low latency (tuning RF higher on LUT fabric might not even save resources, only saves if using DSPs).

&#x09;- Resource strategy: stores weights in BRAM partitioned into RF blocks of size N\_MULT. Each clock cycle, 1 block is read into MAC units -> II = RF clock cycles. Better for large models (won't crash compiler from loop unrolling), higher latency than Latency strategy. Special weight math: power-of-two weights = shifters, binary/ternary weights = AND / XOR gates.

&#x09;- Distributed Arithmetic (DA) strategy: uses external library (da4ml) to turn CMVM into an adder graph (only shift-and-add/subtract, exploits bit-level sparsity). Uses 0 DSPs, lower LUTs and latency than Latency strategy, but ONLY works for RF = 1 (fully unrolled).





* **Parallelization Factor (PF)**: used in Conv layers (lowered to CMVM via im2col). Controls how many CMVM operations happen in parallel in one forward pass (higher PF = lower II, more resources; must divide total CMVM ops evenly).
* Activations:

&#x09;- Piecewise linear (ReLU, Leaky ReLU) -> implemented with multiplexers.

&#x09;- Complex (tanh, sigmoid, softmax) -> pre-computed Look-Up Tables (LUTs) stored in BRAM or LUTRAM (usually \~2,048 entries, 18-bit wide x 1024 deep fits in 1 AMD BRAM; drops LSBs if table size is smaller than input precision).

* Dataflow, Scheduling \& I/O types:

&#x09;- Uses dataflow architecture (#pragma HLS dataflow in Vitis / tasks in oneAPI) -> each layer is a separate hardware function in FPGA fabric running in a pipeline. Higher target clock frequency = compiler inserts more pipeline registers = higher latency in clock cycles.

&#x09;- io\_parallel: default, wires output of one layer directly to input of next layer (partitioned arrays). Max throughput, min latency, for small models.

&#x09;- io\_stream: uses FIFO buffers between layers (shift registers or BRAM). For large models / CNNs where layers have high II. Vivado/Vitis has an RTL co-simulation optimizer that records max FIFO occupancy and shrinks FIFO depth to (max occupation + 1) to save memory.



6.2 IP Core Back Ends

\- Vivado / Vitis (AMD/Xilinx): most complete, supports all layers (dense, conv, RNN, einsum, attention), io\_parallel \& io\_stream, all 3 strategies (Latency, Resource, DA), plus pointwise/depthwise/separable convs.

\- Quartus / oneAPI (Intel/Altera): supports io\_parallel \& io\_stream (oneAPI uses independent tasks connected by pipes/streams). Doesn't have separate Latency/Resource strategies (uses resource-optimized approach by default).

\- Catapult (Siemens EDA): targets ASICs + FPGAs (C++/SystemC). Supports Latency \& Resource, io\_parallel \& io\_stream. Has a "Bottom-Up flow" for streaming -> synthesizes each layer independently into separate RTL / library components so you don't have to re-synthesize the whole model when changing 1 layer (saves server RAM).



6.3 Accelerator Back Ends

\- Wraps the IP core with DMA, memory controllers, host drivers, and multi-threaded host code for full deployment:

&#x09;- Vivado / Vitis IP flow: targets AMD Zynq SoC boards.

&#x09;- Vitis System Design Flow: targets datacenter Alveo cards (overlaps PCIe transfers with computation -> reaches 96% theoretical throughput, >1.4M inferences/sec). Also integrates with Coyote v2 shell.

&#x09;- oneAPI accelerator back end: in development.



7\. Community Tools for Model Optimization and Co-Design

\- 7.1 QKeras: Google QAT library on Keras 2 (QDense, QConv2D; fixed-point, binary, power-of-two) + AutoQKeras for hyperparameter tuning of bit-widths.

\- 7.2 High-Granularity Quantization (HGQ): QAT library (HGQ 1 on Keras 2, HGQ 2 on Keras 3). Differentiable quantization down to per-parameter / sub-layer level (0 bits = automatic pruning). Uses Bit Operations (BOPs) in the loss function (controlled by hyperparameter beta) to estimate hardware resources. Cuts LUTs by 50-95%, DSPs by 50-100%, latency up to 80%, 100% bit-exact in hls4ml.

\- 7.3 da4ml (Distributed Arithmetic): fast compile-time algorithm (takes seconds) that builds adder graphs with common sub-expression elimination for unrolled CMVM. Cuts \~1/3 more LUTs on top of HGQ, uses 0 DSPs, and fixes synthesis crashes on large sparse matrices.

\- 7.4 DSP- and BRAM-Aware Pruning: solves a Knapsack problem (weights get importance value vs hardware cost) to prune whole groups of weights tied to a single DSP or BRAM block. Cuts DSPs by 2.2x-12.2x and BRAM by 1.4x-5.2x.

\- 7.5 Symbolic Regression (SR): translates closed-form math equations (sin, cos, exp from PySR or SymbolNet / SymPy) into HLS hardware via Xilinx math library or LUTs.

\- 7.6 Surrogate Models: fast ML predictors that estimate FPGA resources/latency without waiting hours for HLS synthesis -> rule4ml (MLP trained on 15,000 syntheses, within 10% error) and wa-hls4ml / lui-GNN (GNN trained on >100,000 models).

\- 7.7 Automated Co-Design: NAC (Neural Architecture Co-Design, 2-stage "Once-for-All" search + QAT/pruning), MetaML-Pro (Bayesian optimization across software/hardware stages), plus Bayesian CNNs with Monte Carlo Dropout for uncertainty calibration.

\- 7.8 SoC Integration (ESP): Embedded Scalable Platform wraps hls4ml blocks into tile-based heterogeneous SoCs for ASICs/FPGAs with auto-generated drivers and testbenches.



8\. User-Defined Functionality (Extension API)

\- Lets users add custom unsupported layers (for Keras 2 and PyTorch front ends, works on all back ends) without editing core hls4ml code:

&#x09;1. Custom layer handler function (front end parser).

&#x09;2. Custom layer class inheriting from base Layer in IR.

&#x09;3. Custom Python config/function templates + custom C++ HLS code registered to the back end (and optional custom optimizers).



9\. Evaluation (Benchmarks \& Results)

\- 9.1 High-Level Jet Tagger (16 physics inputs -> 5 classes, MLP):

&#x09;- On AMD XCVU9P: HGQ + DA gets 12.1-23.4 ns latency (3-5 clock cycles at 5ns clock, II=1), 0 DSPs, 0.2%-1.0% LUTs (beats old QKeras baseline of 50-105 ns and thousands of DSPs). Hits >700 MHz Fmax when targeting 1 GHz.

&#x09;- On tiny Artix-7 (XC7A35T): Latency strategy fails placement on largest model, DA strategy fits it (93.6% LUTs, 0 DSPs).

&#x09;- On Altera Agilex7 (oneAPI): QKeras model hits 62 ns (at 341 MHz) to 115 ns (at 782 MHz Fmax), II=1.

\- 9.2 SVHN Classifier (3 Conv + 2 Dense CNN, io\_stream, PF=1 on XCVU9P):

&#x09;- HGQ + DA gets \~93.9% accuracy in 5.2 us (1,045 cycles), 0 DSPs (down from 174 in QKeras), 53k LUTs (down from 111k).

\- 9.3 Particle-Based Jet Tagger (MLP-Mixer, raw data up to 64 particles x 16 features on XCVU13P):

&#x09;- Latency strategy fails to synthesize/meet timing because of huge sparse matrices; HGQ + DA synthesizes cleanly -> 60-65 ns latency (12-13 cycles, II=1), 0 DSPs, 0 BRAM, 3.0%-7.3% LUTs.

\- 9.4 MNIST Classifier (1-layer MLP 768x128 on XCVU13P):

&#x09;- Latency strategy fails loop unrolling; HGQ + DA gets 97.0%-97.7% accuracy in 10 ns (2 clock cycles, II=1), <0.6% LUTs, 0 DSPs.

\- 9.5 Smart Pixels (28 nm CMOS ASIC via Catapult HLS):

&#x09;- On-sensor particle track filter (\~1,200 params in QKeras) -> <1 uW/pixel power, cuts data bandwidth by 54.4%-75.4%.

&#x09;- 5-layer network (depthwise-separable Conv + Dense, \~3,000 params) predicting track angle \& hit position -> 2 clock cycles latency, II=1, takes only 0.81 mm^2 (2D conv) or 0.39 mm^2 (1D conv) chip area.

\- 9.6 Quality of Results (HLS vs Handwritten Verilog \& Versal AIE):

&#x09;- Compared to hand-written Verilog and AMD Versal AI Engines (VLIW processors), hls4ml is often better in latency/resources/throughput (thanks to HLS compiler constant folding) and at worst 2x worse.



10\. Applications of hls4ml

\- CERN LHC (ATLAS \& CMS): filters collision data from 100 TB/s down to 10 GB/s in \~10 us -> deployed MLP for muon trigger, plus AXOL1TL (variational autoencoder) and CICADA (convolutional autoencoder) for real-time anomaly detection.

\- Other Science: cosmic ray radio antenna self-triggering, quantum computing qubit control/readout, Fermilab beam loss monitoring (Arria 10), magnetic fusion plasma mode tracking (>100 kfps, 7.7 us latency), radiation detectors (Artix-7).

\- Industry / Space / Bio: self-driving car segmentation (\~5 ms on ZCU102), satellite Earth/marine monitoring (Edge SpAIce), food contamination microwave sensing (27 us on Kria K26), artificial surface texture C-RNN (6.21 us), ECG arrhythmia 1D-CNN (1.655 W), lymphocyte cell sorting (12x faster than GPU), 100 Gbps network packet malware filter (ternary net, 44 ns latency), speech keyword spotting (6x-12x faster than CPU/GPU).



11\. Challenges and Limitations

\- Huge maintenance burden supporting all combinations of front ends x back ends (uneven feature support across back ends, research code is hard to generalize back into main repo).

\- HLS compiler limits: very large models cause multi-hour compile times or synthesis crashes due to on-chip unrolling/congestion.

\- External tool changes: compiler versions change pragmas/heuristics (requiring template re-tuning), or vendors drop tools completely (like Intel dropping Altera FPGAs in oneAPI 2025.1).



12\. Related Work

\- Two types of FPGA accelerators: (1) Software-controlled Overlays (fixed microarchitecture like CPU/GPU: AMD Vitis AI, Intel FPGA AI Suite, EdgeCortix MERA - don't support multi-vendor or arbitrary quantization) vs (2) Custom Hardware Generators (model-specific circuits like hls4ml).

\- Other custom generators: FP-DNN, DeepBurning, DeepHLS, PipeCNN, DnnWeaver, NNgen (unmaintained, single framework/arch); fpgaConvNet \& HPIPE/H2PIPE (CNN-focused, H2PIPE uses off-chip memory); MASE (PyTorch SystemVerilog compiler for Transformers/LLMs); CGRA4ML (reuses PUs + off-chip memory for large models, QKeras only); chisel4ml (Chisel HDL from QONNX); tinyHLS (Verilog for Keras 2 MLP/1D-CNN).

\- FINN (AMD/Xilinx): closest to hls4ml (dataflow, arbitrary quantization via Brevitas/QONNX), but only targets AMD/Xilinx FPGAs, whereas hls4ml supports multiple ML frameworks + multiple FPGA/ASIC vendors.



13\. Conclusions and Outlook (Future Work)

\- New front/back ends in development: Flax/JAX front end, SmartHLS back end (Microchip PolarFire FPGAs), radiation-tolerant NanoXplore FPGAs (via Catapult or PandA Bambu HLS toolchain).

\- New architectures \& scaling: expanding Transformer support (including HEPT - locality sensitive hashing point transformer), integrating AMD Versal AI Engines (AIEs), and distributed multi-FPGA inference using Coyote networking shell for large models that don't fit on 1 chip.











\[1] Compiler \[C++ --> RTL (verilog ou vhdl)] : **Environement** -> (Siemens, complet)

\[2] Siemens EDA (formerly Mentor Graphics, acquired by Siemens in 2017) provides a comprehensive suite of software tools used to design, verify, and manufacture electronic circuits, specifically Integrated Circuits (ICs), ASICs, and printed circuit boards (PCBs).

