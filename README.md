# Self-Improving Language Model Demo

This is a simplified implementation of a self-improving language model that can generate its own training data, fine-tune itself, and evaluate its performance in an iterative loop.

## Features

- **Self-Generating Training Data**: Creates synthetic training examples based on a task description
- **Fine-Tuning**: Adapts the model to the specific task through fine-tuning
- **Evaluation**: Assesses model performance on test examples
- **Iterative Improvement**: Repeats the process to continuously improve performance

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the demo with default settings:

```bash
python self_improving_lm.py
```

### Customizing the Task

Edit the `main()` function in `self_improving_lm.py` to change the task description:

```python
task = "Your task description here"
```

## How It Works

1. **Initialization**: Loads a pre-trained language model (GPT-2 by default)
2. **Self-Improvement Loop**:
   - Generates training data based on the task
   - Fine-tunes the model on the generated data
   - Evaluates performance on test examples
   - Repeats the process for the specified number of iterations

## Example Output

```
Starting self-improvement for task: Answer simple arithmetic questions with just the number.

--- Iteration 1/2 ---
Generating training data...
Fine-tuning the model...
Evaluating model...

Evaluation Results (Iteration 1):
  True: 7 | Predicted: 7
  True: 42 | Predicted: 42
  Accuracy: 100.0%

--- Iteration 2/2 ---
...
```

## Limitations

This is a simplified demo. In a production system, you would want to:
- Use a larger base model
- Implement more sophisticated data generation
- Use a separate validation set
- Implement early stopping
- Add more comprehensive evaluation metrics

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- scikit-learn
- tqdm
- NumPy
