# Chapter 11: Recurrent Neural Networks and Long Short-Term Memory

## Chapter Overview
Standard feedforward networks and CNNs assume fixed-size inputs and lack the ability to natively process temporal ordering. This chapter explores sequence modeling through recurrent architectures. We analyze the mathematical origins of the vanishing and exploding gradient problems in vanilla RNNs and demonstrate how Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) resolve them via constant error carousels.

## Learning Objectives
* Unroll the recurrent computation graph and derive Backpropagation Through Time (BPTT).
* Mathematically prove why vanilla RNNs suffer from vanishing gradients when the recurrent weight eigenvalues and activation derivatives multiply to less than 1.0[cite: 1].
* Implement LSTM and GRU cells, understanding the explicit equations for the forget, input, and output gates[cite: 1].
* Build a Sequence-to-Sequence (Seq2Seq) model with Bahdanau (additive) attention to resolve the fixed-length context bottleneck[cite: 1].

## Concepts Covered
* BPTT and Gradient Clipping
* The Vanishing / Exploding Gradient Problem
* Long Short-Term Memory (LSTM)
* Gated Recurrent Units (GRU)
* Bidirectional RNNs
* Seq2Seq and Bahdanau Attention

## Connection to Textbook
The implementations mirror the exact mathematical structures detailed in Chapter 11. The attention mechanism explicitly calculates the context vector as the weighted sum of encoder hidden states, directly using the energy equations from the text[cite: 1].
