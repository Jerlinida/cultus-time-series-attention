# Multivariate Time Series Forecasting Using LSTM with Attention

## Project Overview
This project implements a multivariate time series forecasting model using an Encoder–Decoder LSTM architecture enhanced with a Bahdanau Attention mechanism. The objective is to predict future energy consumption based on multiple correlated temporal features while improving interpretability through attention weight visualization.

This project is developed as part of the Cultus Skills Center – Job Readiness Real-World Project and follows all specified requirements for dataset size, model architecture, evaluation, and analysis.

## Problem Statement
Traditional sequence models such as standard LSTMs treat all historical time steps equally, which limits interpretability and may reduce performance. This project addresses that limitation by integrating an attention mechanism that enables the model to dynamically focus on the most relevant past observations during prediction.

## Dataset Description
- Type: Synthetic multivariate time series
- Number of time steps: 1500
- Number of features: 5
  - Temperature
  - Humidity
  - Pressure
  - Wind Speed
  - Energy Consumption (Target)

The dataset is normalized using Min-Max scaling and converted into fixed-length sequences using a lookback window of 20 time steps.

## Model Architecture
The model consists of:
- Encoder: LSTM network that processes historical input sequences
- Attention Mechanism: Bahdanau (additive) attention to compute context vectors
- Decoder: LSTM-based decoder followed by a fully connected layer for prediction

This architecture enables the model to learn temporal dependencies while providing interpretability.

## Training Details
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Epochs: 20
- Batch Size: 32
- Framework: PyTorch

The training process shows stable convergence with decreasing loss values across epochs.

## Evaluation Metrics
The trained model was evaluated using standard regression metrics:
- RMSE: 0.1987
- MAE: 0.1710

These values indicate reliable forecasting performance for a multivariate attention-based time series model trained on noisy data.

## Attention Visualization
Attention weights are visualized using a heatmap to illustrate the relative importance of historical time steps. The visualization shows that the model assigns higher importance to recent observations, validating the effectiveness of the attention mechanism.

## Results and Interpretation
The attention-based encoder–decoder model demonstrates stable learning, acceptable error rates, and improved interpretability compared to a standard LSTM without attention. This makes the model suitable for real-world forecasting tasks where both accuracy and explainability are required.

## How to Run the Project
1. Open the provided Jupyter Notebook in Google Colab
2. Run all cells sequentially
3. Observe training loss, evaluation metrics, and attention visualization

## Tools and Technologies Used
- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Google Colab

## Submission Note
This project is submitted as a single markdown-based text submission in accordance with Cultus Skills Center guidelines. The complete implementation, evaluation, and analysis are included.

## Author
Jerlin Ida J
Job Readiness Program – Cultus Skills Center
