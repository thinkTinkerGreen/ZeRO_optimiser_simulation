---
authored_by: human
author_name: "<filled by human>"
date: "<YYYY-MM-DD>"
ai_assistance_statement: "<human states exactly how agents were and were not used>"
---

# My Understanding of ZeRO Optimisation

### Q1 Derive 16Ψ for mixed-precision Adam: where does each term come from, and why is K = 12?
<!-- HUMAN: write in your own words -->

### Q2 Why does stage 1 leave communication volume unchanged? Walk through reduce-scatter + all-gather = all-reduce.
<!-- HUMAN: write in your own words -->

### Q3 Using your own memory-timeline figure, explain exactly what differs between stage 1 and stage 2 and when in the step it happens.
<!-- HUMAN: write in your own words -->

### Q4 Why is stage 3 3Ψ rather than 2Ψ? Name each collective and when it fires.
<!-- HUMAN: write in your own words -->

### Q5 What does this simulator model, and what does it ignore compared with real GPUs?
<!-- HUMAN: write in your own words -->

### Q6 Pick one number in your results that surprised you and trace it to the code that produced it.
<!-- HUMAN: write in your own words -->

### Q7 What is padding overhead here, why does it exist, and how big is it for your configs?
<!-- HUMAN: write in your own words -->

### Q8 Why is the float64 equivalence test the right correctness test? Describe what a stage-2 bug would look like in it.
<!-- HUMAN: write in your own words -->

### Q9 Activations did not shrink across stages. Why, and what does ZeRO-R do about it?
<!-- HUMAN: write in your own words -->

### Q10 Map your stages onto FSDP2 and DeepSpeed settings. What is the same, what is different?
<!-- HUMAN: write in your own words -->

### Q11 Given your numbers, when would you pick stage 2 over stage 3?
<!-- HUMAN: write in your own words -->

### Q12 Describe a bug or wrong assumption you hit while building this and how you found it.
<!-- HUMAN: write in your own words -->
