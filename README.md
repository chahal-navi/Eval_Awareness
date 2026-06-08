# Eval_Awareness **in Large Language Models**

Ongoing research project. Manuscript in preparation.
## Overview
This repository contains code and data for an interpretability study investigating how large language models detect that they are being evaluated. We examine whether models at the ~3B parameter scale exhibit evaluation-aware behavior, and what internal mechanisms drive that behavior.
The project focuses on Phi-3 3.8B and uses mechanistic interpretability methods to move beyond behavioral observation toward causal understanding of evaluation awareness.
Research Questions

1. Do models at this scale exhibit systematic behavioral differences between evaluation and deployment contexts?
2. What structural and semantic features of prompts trigger evaluation-aware behavior?
3. Where in the model's computational graph does evaluation awareness originate?

## Methodology
### Dataset Construction
We construct contrastive prompt pairs across three categories designed to isolate distinct potential triggers of evaluation awareness:

Task Formatting — prompts that vary evaluation-typical structural cues while holding content constant
Synthetic Inputs — prompts that vary the naturalness and origin of the input
Memorization — prompts that vary exposure to evaluation-typical content during pretraining

Each category produces matched pairs where one prompt is evaluation-typical and one is deployment-typical, with all other variables controlled.

### Probing
Linear probes are trained on residual stream activations at each layer to detect evaluation-awareness signals. Probe accuracy across layers identifies where evaluation-relevant information is represented in the model.
### Feature Attribution
Prompt-level Integrated Gradients (IG) attribution identifies which tokens and structural features drive evaluation-aware activations, providing interpretable explanations of what the model is responding to.
### Causal Verification
Activation patching verifies that identified features causally produce evaluation-aware behavior, rather than merely correlating with it. Out-of-distribution generalization tests assess robustness of the identified mechanisms.

## Repository Structure
data/                  
notebooks/             
  01_probe_training
  02_ig_attribution
  03_activation_patching
figures/                
results/               
## Selected Results
<img width="898" height="838" alt="Screenshot 2026-06-09 at 12 32 17 AM" src="https://github.com/user-attachments/assets/2e1f9ed1-bf23-4f3a-b05e-c11e6e54bf11" />

# Prompt Editing Pipeline
As a downstream application of the IG attribution results, we built a pipeline that automatically edits prompts to reduce evaluation-awareness eliciting features. The pipeline uses the dataset of high-attribution phrases identified by IG to detect and neutralize structural and semantic triggers in arbitrary input prompts, producing edited versions that elicit more deployment-typical model behavior. This provides a practical intervention tool and serves as an additional causal check — if edited prompts systematically reduce evaluation-aware activations, it validates that the identified features are genuinely driving the behavior rather than correlating with it.
<img width="1441" height="635" alt="Screenshot 2026-06-09 at 12 54 20 AM" src="https://github.com/user-attachments/assets/05e633d2-2638-40db-8198-855b9a8b84ea" />
### Status
Active research project. Code is research-grade and partially cleaned. Notebooks are numbered in suggested execution order. Dataset generation script produces the full contrastive dataset from scratch.
Manuscript in preparation. Please contact before building directly on this work. ms22110@iisermohali.ac.in
