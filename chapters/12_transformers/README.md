# Chapter 12: The Transformer Architecture

## Chapter Overview
This chapter details the paradigm shift from recurrent networks to purely attention-based architectures. By relying entirely on self-attention mechanisms, Transformers achieve $O(1)$ path lengths between sequence positions and enable massive parallelization. We dissect the multi-head attention mechanism and explore landmark models including BERT and GPT.

## Learning Objectives
* Build the Scaled Dot-Product Attention mechanism and understand the requirement for the $\sqrt{d_k}$ scaling factor[cite: 1].
* Implement Multi-Head Attention to project queries, keys, and values into parallel representational subspaces[cite: 1].
* Generate Sinusoidal Positional Encodings to inject sequential order into permutation-invariant attention layers[cite: 1].
* Differentiate the Masked Language Modeling (MLM) of BERT from the Autoregressive causal modeling of GPT[cite: 1].

## Concepts Covered
* Key-Query-Value Abstraction
* Scaled Dot-Product and Multi-Head Attention
* Sinusoidal Positional Encoding & RoPE
* Pre-Norm vs. Post-Norm Residual Architectures
* Bidirectional Encoders (BERT)
* Autoregressive Decoders (GPT)
* Neural Scaling Laws

## Connection to Textbook
The custom from-scratch implementations provided here are mathematically equivalent to the attention formulas $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$[cite: 1]. The `TransformerEncoderLayer` reflects the exact Pre-Norm architecture favored by modern Large Language Models as discussed in the text[cite: 1].
