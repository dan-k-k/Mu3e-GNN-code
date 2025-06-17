# Graph Neural Networks for Track Finding in Mu3e
> **Dan King**  

The Mu3e experiment at the Paul Scherrer Institute searches for the charged-lepton-flavour-violating decay μ⁺ → e⁺ e⁺ e⁻ (signal) with the goal of excluding branching fractions above 10<sup>−16</sup> at the 90% confidence level. This decay is essentially forbidden by the Standard Model (SM) and any observation would provide evidence of new physics. The experiment faces a loss in the number of observed MC signal decays (i.e. a lowered efficiency) when applying cuts that purify signal detection. An alternative approach involves a Graph Neural Network (GNN) and finds signal and beam efficiencies similar to the official standard reconstruction algorithm. More information on the experiment is available in the [Technical design of the phase I Mu3e experiment](https://arxiv.org/abs/2009.11690) by the Mu3e collaboration.

<p align="center">
  <img
    src="images/Design_Recurling_tracks.png"
    alt="Mu3e TDR Detector"
    width="73%"
  />
</p>

This repository contains the full implementation of a Graph Neural Network (GNN) model for the Mu3e experiment:

- Implements a six‐hit graph-building procedure with loose geometric and kinematic constraints, pruning away most combinatorial (fake) graphs.
- Defines and trains a three-class GNN (electron vs. positron vs. fake) using PyTorch Geometric, including custom message-passing layers, batch-norm, pooling, and truth data concatenation.

- Process:
  - Load Mu3e MC hits and construct node/edge/global features for each six‐hit candidate graph.
  - Normalise all graph features using global means/σ from the signal set.
  - Split into train/val/test, early-stopping on validation loss, and save the best model.

- Evaluation:
  - Compute per-class ROC/AUC, purity, efficiency, and Clopper–Pearson errors.
  - Deduplicate true (real) graphs due to overlapping Mu3e detection ladders.
  - Perform greedy overlap-removal (hit-sharing) among predicted real graphs to improve purity at the cost of efficiency.

- Helper scripts:
  - Create Pandas DataFrames from deduplicated graphs (with model predictions and all truth/feature information) for analysis.
  - Plot confidence histograms, purity and efficiency vs. truth info and hit-counts per frame.

### Data Availability

The original Mu3e Monte Carlo “.root” files used to build the graphs are **not** included in this repository. 

## How to run

1. Clone:  
   ```bash
   git clone https://github.com/dan-k-k/Mu3e-GNN-code.git
   cd Mu3e-GNN-code
   ```

2. Create virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install requirements:
   ```
   pip install -r requirements.txt
   ```

4. Generate graphs through the `Mu3e_Graph_Generation.ipynb` notebook

5. Train the model and view results using the `Mu3e_GNN_and_Results.ipynb` notebook

## Report overview:

Monte Carlo simulations accurately model the expected decay particles' hits left in the detector, made up of frames: a period of time long enough to capture all hits from the decay. The Graph Neural Network approach contained in this repository focuses on six-hit tracks that reach the recurl pixel layers. A three-class GNN uses spatial and kinematic information of combinations of hits left in the detector to classify each six-hit graph as an electron, positron, or fake (combinatorial background). Common background decay types are internal conversion μ⁺ → e⁺ e⁺ e⁻ ν<sub>e</sub> ν̄<sub>μ</sub> and Michel μ⁺ → e⁺ ν<sub>e</sub> ν̄<sub>μ</sub>. 

<p align="center">
  <img
    src="images/nparticlesperframebeam2.png"
    alt="Number of particles per frame"
    width="26%"
  />
  <img
    src="images/nhitssperframebeam2.png"
    alt="Number of hits per frame"
    width="32%"
  />
</p>

Graphs are formed by combining hits that follow a detection layer sequence, assuming no track information is known. A cheap constraints algorithm for graph generation greatly reduces processing time, essentially focusing on graphs that closest resemble true particle tracks. 

<p align="center">
  <img
    src="images/ngeneratedgraphs1.png"
    alt="Mu3e TDR Detector"
    width="30%"
  />
</p>

The GNN model is trained and validated on a dataset of ~92,000 six-hit graphs of one signal decay event per frame. In the beam simulation, at most one signal decay is expected to be seen in a frame among the background decays.

<p align="center">
  <img
    src="images/trainingvalidationloss25apr.png"
    alt="Training and Validation Loss Curve"
    width="35%"
  />
</p>

The model returns outputs for each class on test graphs (for signal, this is 20% of the available set). The outputs are the model's estimated probabilities of the graph belonging to each class. In some cases, the model will incorrectly give true electron and positron graphs low outputs and vice versa. Note that the y axis is logarithmic.

<p align="center">
  <img
    src="images/GNNoutputsignal1.png"
    alt="GNN Output: Signal"
    width="32%"
  />
</p>

Electron/positron misclassification is negligible.

<p align="center">
  <img
    src="images/electronseparationsignal.png"
    alt="Electron Separation: Signal"
    width="31%"
  />
  <img
    src="images/electronseparationIC.png"
    alt="Electron Separation: Internal Conversion"
    width="31%"
  />
</p>

There is a trade-off between purity and efficiency. More real and fake graphs are classified real as the required score to pass is lowered. This increases efficiency (more of the true real graphs are correctly recalled), but lowers purity (more fakes are among the real predictions).

<p align="center">
  <img
    src="images/purityeffsignalelecpos.png"
    alt="Purity against Efficiency: Signal"
    width="32%"
  />
</p>

For frames with more hits, the model's performance is worse as there is higher chance of nearby hits generating fakes that closely resemble real tracks. For frames with equal hits, the beam set tends to have worse purity due to individual Michel decay events with uniform spatial distributions as opposed to, for example, signal, requiring all electrons to conserve momentum. 

<p align="center">
  <img
    src="images/purityvsnhits2.png"
    alt="Purity against the Number of Hits per Frame"
    width="28%"
  />
  <img
    src="images/purityvsngraphs2.png"
    alt="Purity against the Number of Generated Graphs per Frame"
    width="28%"
  />
</p>

Comparing the full GNN method efficiencies for all datasets with the official standard reconstruction and Clopper-Pearson one-sigma errors (with a GNN-approach beam purity of 94.5% and signal purity of 97.7%, while the official standard reconstruction approach is considered pure).

<p align="center">
  <img
    src="images/comparisoneffCP2.png"
    alt="Efficiencies with Clopper-Pearson one-sigma Errors"
    width="49%"
  />
</p>
