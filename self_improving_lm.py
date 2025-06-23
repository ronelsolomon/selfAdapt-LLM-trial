import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    is_tf_available,
    set_seed
)
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random
from typing import List, Dict
import logging
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
set_seed(42)

# Disable TensorFlow GPU visibility
if is_tf_available():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel(logging.ERROR)

# Logger setup
logging.basicConfig(level=logging.INFO)

class SelfImprovingLM:
    def __init__(self, base_model_name: str = "distilgpt2"):
        """Initialize the self-improving language model."""
        self.model_name = base_model_name
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded: {base_model_name}")
        print(f"Model device: {next(self.model.parameters()).device}")

    def generate_training_data(self, task_description: str, num_examples: int = 50) -> List[Dict]:
        """Generate synthetic arithmetic data."""
        data = []
        for _ in range(num_examples):
            a, b = random.randint(1, 20), random.randint(1, 20)
            op = random.choice(["+", "-", "*", "/"])
            if op == "/" and b == 0:
                continue
            expression = f"{a}{op}{b}"
            try:
                result = str(int(eval(expression)))
                prompt = f"Task: {task_description}\nInput: What is {expression}?"
                data.append({"input": prompt, "output": result})
            except:
                continue
        return data

    def fine_tune(self, training_data: List[Dict], num_epochs: int = 3):
        """Fine-tune the model on synthetic data."""
        class CustomDataset(Dataset):
            def __init__(self, data, tokenizer):
                self.data = data
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                text = f"{item['input']} Output: {item['output']}{self.tokenizer.eos_token}"
                encoding = self.tokenizer(
                    text,
                    max_length=128,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": encoding["input_ids"].squeeze().clone()
                }

        dataset = CustomDataset(training_data, self.tokenizer)

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            save_steps=500,
            save_total_limit=1,
            logging_dir='./logs',
            logging_steps=5,
            disable_tqdm=False,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

    def evaluate(self, test_data: List[Dict]) -> float:
        """Evaluate model performance."""
        self.model.eval()
        predictions, true_labels = [], []

        for item in test_data:
            try:
                input_text = item["input"]
                true_output = item["output"]

                inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)

                with torch.no_grad():
                    output = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=inputs["input_ids"].shape[1] + 10,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=False
                    )

                output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                prediction = output_text.split("Output:")[-1].strip().split()[0]
                predictions.append(prediction.strip())
                true_labels.append(true_output.strip())
            except Exception as e:
                print(f"Evaluation error: {e}")

        accuracy = accuracy_score(true_labels, predictions) if predictions else 0.0
        return accuracy, list(zip(true_labels, predictions))

    def self_improve_loop(self, task_description: str, num_iterations: int = 3):
        """Self-improving training loop."""
        print(f"Starting self-improvement for task: {task_description}")

        for i in range(num_iterations):
            print(f"\n--- Iteration {i+1}/{num_iterations} ---")

            training_data = self.generate_training_data(task_description, num_examples=50)
            print(f"Training examples generated: {len(training_data)}")

            self.fine_tune(training_data)

            test_data = [
                {"input": f"Task: {task_description}\nInput: What is 3+4?", "output": "7"},
                {"input": f"Task: {task_description}\nInput: What is 6*7?", "output": "42"},
                {"input": f"Task: {task_description}\nInput: What is 12-5?", "output": "7"},
                {"input": f"Task: {task_description}\nInput: What is 8/2?", "output": "4"},
            ]

            accuracy, results = self.evaluate(test_data)

            print(f"\nEvaluation Results (Iteration {i+1}):")
            for true, pred in results:
                print(f"  True: {true} | Predicted: {pred}")
            print(f"  Accuracy: {accuracy * 100:.1f}%")

            # Save model and tokenizer for next iteration
            iter_dir = f"./checkpoint_iter_{i+1}"
            self.model.save_pretrained(iter_dir)
            self.tokenizer.save_pretrained(iter_dir)
            print(f"  Saved model to {iter_dir}")

            # Reload updated model
            self.model = AutoModelForCausalLM.from_pretrained(iter_dir).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(iter_dir)
            self.tokenizer.pad_token = self.tokenizer.eos_token


def main():
    model = SelfImprovingLM()
    task = "Answer simple arithmetic questions with just the number."
    model.self_improve_loop(task, num_iterations=2)

    # Save final model
    output_dir = "./self_improved_model"
    model.model.save_pretrained(output_dir)
    model.tokenizer.save_pretrained(output_dir)
    print(f"\nFinal model saved to {output_dir}")


if __name__ == "__main__":
    main()

