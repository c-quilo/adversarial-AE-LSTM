# adversarial-AE-LSTM
Using adversarial training to improve forecasts of data-driven surrogate models of CFD simulations

Data-driven approaches can be seen as attractive solutions to produce reduced-order models (ROMs) of Computational Fluid Dynamics (CFD) simulations. Moreover, forecasts produced by ROMs are obtained at a fraction of the cost of the original CFD model solution when used together with a ROM. Recurrent neural networks (RNN) have been used to model and predict temporal dependencies between inputs and outputs of ROMs. Non-intrusive ROMs and RNNs have been used together in previous studies where the surrogate forecast systems can easily reproduce a time-step in the future accurately. However, when the predicted output is used as an input for the prediction of the subsequent time sequence, the results can detach quickly from the underlying physical model solution when encountering out-of-distribution data.

Our framework relies on adversarial training to improve the longevity of the surrogate forecasts and it consists of two steps:

1. Two-step dimension-reduction: Firstly, Principal Component Analysis (PCA) is used to reduce the dimensions of the system (extractFieldsAndPCA). Secondly, an adversarial AE is trained on the Principal Components of the model solution in order to create a latent space with a normal distribution (adversarial_PCAE).

2. The latent space is then used to train an adversarial LSTM (). Robustness which may be achieved by detecting and rejecting adversarial examples by using adversarial training, allowing us to reduce the divergence of the forecast prediction over time and better compression from full-space to latent space.

Statistics of the model can be plotted using () and ().
