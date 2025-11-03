# ZSSC: Zero-Shot Approach to Overcome Perturbation Sensitivity of Prompts

[cite_start]This repository contains a project implementation of the research paper: **"Zero-shot Approach to Overcome Perturbation Sensitivity of Prompts"** (ZS-SC)[cite: 3].

[cite_start]The project focuses on improving the performance of prompts for binary sentence-level sentiment classification in a **zero-shot setting**—that is, without relying on any labeled training data[cite: 10].

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-blue)](https://huggingface.co/transformers/)

## 📖 About the Project

[cite_start]Language models are highly sensitive to the specific wording (perturbations) of the prompts they are given[cite: 8]. [cite_start]Finding the "perfect" prompt often requires extensive manual effort or few-shot learning, which depends on labeled examples[cite: 7, 9].

This project implements the **ZS-SC** method, which tackles this problem by:
1.  [cite_start]**Augmenting** a single, simple "base prompt" into many different candidate prompts[cite: 11].
2.  [cite_start]**Ranking** these candidate prompts using a novel metric that requires **no labeled data**[cite: 11].
3.  [cite_start]**Selecting** the top-ranked prompt to perform high-accuracy, zero-shot sentiment classification[cite: 12].

## 🧠 How ZS-SC Works

The ZS-SC method is a two-stage process: **Prompt Augmentation** and **Zero-Shot Ranking**.


### 1. Prompt Augmentation

[cite_start]Given a single base prompt (e.g., "The sentence was [MASK]") [cite: 11, 274][cite_start], the method automatically generates a large set of diverse candidate prompts using three techniques[cite: 40, 141]:

* [cite_start]**Positioning:** Places the prompt either before or after the input sentence[cite: 142].
* [cite_start]**Subordination:** Uses conjunctions like "so" and "because" to create a dependency between the sentence and the prompt[cite: 143].
* [cite_start]**Paraphrasing:** Uses a masked language model (like BERT) to predict and swap tokens within the prompt, creating semantically similar variations (e.g., "The sentence was..." → "This feedback is...")[cite: 163, 165].

### 2. Novel Zero-Shot Ranking

This is the core contribution of the paper. To rank the augmented prompts without labeled data, ZS-SC uses a novel metric based on a key intuition:

> [cite_start]**A high-quality prompt should be sensitive to changes in sentiment-carrying keywords**[cite: 42, 339].

The metric works by:
1.  [cite_start]Taking an unlabeled sentence from the corpus that contains a strong sentiment word (e.g., "Battery life was **great**")[cite: 198, 204].
2.  **Flipping the Polarity:** Replacing "great" with its opposite (e.g., "terrible"). [cite_start]A good prompt should **flip** its predicted label[cite: 195, 198].
3.  **Using a Synonym:** Replacing "great" with a synonym (e.g., "excellent"). [cite_start]A good prompt should **keep** its predicted label the same[cite: 197, 199].

[cite_start]Prompts are scored based on how consistently they behave this way across the unlabeled dataset[cite: 218]. [cite_start]WordNet is used to generate a rich set of synonyms for this process[cite: 210].

### 3. Prediction

[cite_start]Finally, the prompt with the highest score (the top-1 prompt) is selected to perform sentiment classification on the entire dataset[cite: 231]. [cite_start]The paper also explores aggregating the predictions of the top-k prompts[cite: 233, 235].

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* PyTorch
* Hugging Face Transformers
* NLTK (for WordNet)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/RushikeshSonwane03/NLP_Project-ZSSC.git](https://github.com/RushikeshSonwane03/NLP_Project-ZSSC.git)
    cd NLP_Project-ZSSC
    ```

2.  **Install dependencies:**
    (It's recommended to use a virtual environment)
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download NLTK data:**
    Run the following in a Python interpreter:
    ```python
    import nltk
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    ```

## ⚙️ Usage

You can run the full ZS-SC pipeline (augmentation, ranking, and evaluation) from the main script.

*(Note: Please modify the command below to match the exact arguments in your `main.py` script)*

```bash
python main.py \
    --model_name "bert-base-uncased" \
    --dataset "SST-2" \
    --base_prompt "The sentence was [MASK]" \
    --mapping "{'positive': 'great', 'negative': 'terrible'}" \
    --k_paraphrase 30 \
    --top_k_aggregate 1
```

## 📊 Results

[cite_start]The goal of this implementation is to replicate the paper's findings, which demonstrate that ZS-SC's top-ranked prompts significantly outperform both the original base prompt and other baseline methods on benchmark datasets like **SST-2, MR, and CR**[cite: 12, 278].

[cite_start]The paper shows that this zero-shot method can even outperform few-shot learning techniques, proving the effectiveness of the augmentation and ranking strategy[cite: 12].

## 📜 Reference

This project is an implementation of the following paper. All credit for the method and concepts goes to the original authors.

[cite_start]**Title:** Zero-shot Approach to Overcome Perturbation Sensitivity of Prompts [cite: 3]
[cite_start]**Authors:** Mohna Chakraborty, Adithya Kulkarni, and Qi Li [cite: 4]
[cite_start]**arXiv:** [2305.15689v2 [cs.CL]](https://arxiv.org/abs/2305.15689v2) [cite: 1]

```bibtex
@misc{chakraborty2023zeroshot,
  title={Zero-shot Approach to Overcome Perturbation Sensitivity of Prompts},
  author={Mohna Chakraborty and Adithya Kulkarni and Qi Li},
  year={2023},
  eprint={2305.15689},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
