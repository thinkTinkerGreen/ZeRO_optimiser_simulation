---
authored_by: Self
date: 2026-10-23
---

# My Understanding of ZeRO Optimisation

### Q1 Derive 16Ψ for mixed-precision Adam: where does each term come from, and why is K = 12?
The 16Ψ memory footprint represents the total bytes required to store a model's states during mixed-precision training with the Adam optimizer, where Ψ represents the number of parameters in the model.
The model states are: Parameters - FP16 and Gradients - FP16 (2 each). Total 4Ψ. 
Adam optimiser maintains three components in FP32 (Master Parameters, Momentum, and Variance)  4 each. Total 12Ψ. This is why we have 16Ψ total.
These three components of the optimiser are maintained in FP32, so we use K = 12 for the optimiser states. 

### Q2 Why does stage 1 leave communication volume unchanged? Walk through reduce-scatter + all-gather = all-reduce.

In stage 1, we partition the optimiser states across the data parallel ranks. This means we can perform reduce-scatter + all-gather to replicate the full set of optimiser states across all ranks. Each GPU keeps the Master parameters and optimiser averages for one weight it owns. It also holds the weight and gradients for all 8 weights 

### Q3 Using your own memory-timeline figure, explain exactly what differs between stage 1 and stage 2 and when in the step it happens.
The difference between a stage 1 and stage 2 is that in stage 1, we keep all the gradients on each GPU and perform all reduce to update the weights. In stage 2, we partition the gradients across the data parallel ranks and perform reduce-scatter + all-gather to update the weights. This reduces the memory usage by a factor of N. It happends in the backward pass.  

### Q4 Why is stage 3 3Ψ rather than 2Ψ? Name each collective and when it fires.
Stage 3 is 3Ψ rather than 2Ψ because we have to all-gather the parameters and the gradients across the data parallel ranks. All gather takes place in the forward pass to gather the parameters for the current layer, and then all gather takes place in the backward pass to gather the gradients for the current layer. Finally we perform reduce-scatter to update the weights. So that is 2Ψ for all-gather (once in the forward pass and once in the backward pass ) and  Ψ for reduce-scatter (once in the backward pass ). 

### Q5 What does this simulator model, and what does it ignore compared with real GPUs?
This simulator model creates 1D parallelism across the data parallel ranks. It ignores the fact that the model is actually 3D parallelism, i.e. we can have model parallelism and pipeline parallelism as well. It also ignores the fact that we can have 2.5 D parallelism, i.e. we can have model parallelism and pipeline parallelism as well. 

It does not consider communication overlap between the different optimiser states. 

### Q6 What are the key differences between stage1, stage 2, stage 3, and stage3-ZeRO R?
• Stage 1: Partitions only the optimizer states (such as Adam's 32-bit master weights and momentum terms) across GPUs. Model parameters and gradients remain fully replicated.
• Stage 2: Partitions both the optimizer states and the gradients across GPUs. Model parameters still remain fully replicated, meaning each GPU holds a full copy of the weights at all times.
• Stage 3: Partitions optimizer states, gradients, and 16-bit model parameters across all data-parallel ranks. This means each GPU only holds 1/N of the model.
• Stage 3-ZeRO-R: Same as Stage 3, but with overlapping communications. This means that the communication between the GPUs is overlapped with the computation, which reduces the communication time.

### Q7 What is padding overhead here, why does it exist, and how big is it for your configs?
The padding overhead is the extra memory that is used to store the model parameters and gradients and optimiser states. It exists because the model parameters and gradients and optimiser states are not always divisible by the number of ranks. They are padded with dummy values to satisfy division or hardware alignment requirements

### Q8 Why is the float64 equivalence test the right correctness test? Describe what a stage-2 bug would look like in it.
This test compares the output of the model with the output of the model with the optimiser in FP32 or FP16 but without the numerical errors that can occur using mixed precision. A stage-2 bug would look like the output of the model with the optimiser in FP16 being different from the output of the model with the optimiser in FP32. For example if we have a small learning rate or small model, the gradient would be very small. So when the FP16 gradients are all-reduced, the small gradients would be lost due to rounding and underflow errors.  

### Q9 Activations did not shrink across stages. Why, and what does ZeRO-R do about it?
The memory used by the activations is independent of the stage of the ZeRO optimiser. This is because the activations are not partitioned across the data parallel ranks. They are instead stored on the device that computes them. ZeRO-R does not address this, but there are other techniques that can be used to reduce the memory usage of the activations, such as activation checkpointing. 

### Q10 Given your numbers, when would you pick stage 2 over stage 3?
When the model size is small or the number of GPUs is large, we should use stage 2 instead of stage 3 because it is faster and uses less memory. But when the model size is large or the number of GPUs is small, we should use stage 3 because it is faster and uses less memory. In our case we have model size of 1 billion parameters and 8 GPUs. So we should use stage 2 instead of stage 3 because it is faster and uses less memory.



