# Comparative LLM Fine-Tuning for Banking77 Intent Classification

## 1. Project Overview

This project compares three approaches for classifying banking-related customer queries into 77 intent categories using the Qwen2.5-0.5B-Instruct language model:

1. Pretrained prompt-based baseline
2. Parameter-efficient fine-tuning using LoRA
3. Full fine-tuning

The project evaluates classification performance and considers the trade-off between model quality, trainable parameters, and computational resources.

## 2. Dataset

- **Dataset:** Banking77
- **Source:** https://huggingface.co/datasets/DeepPavlov/banking77
- **Task:** Single-label intent classification
- **Number of intent classes:** 77
- **Total examples:** 13,083

### Data Partitions

| Partition | Examples | Purpose |
|---|---:|---|
| Training | 9,002 | Model training |
| Validation | 1,001 | Validation and model selection |
| Test | 3,080 | Final evaluation |

The original training partition was split into training and validation subsets using stratification and random seed 42. The official test partition was retained for final evaluation.

## 3. Model and Methods

- **Base model:** Qwen/Qwen2.5-0.5B-Instruct
- **Baseline:** Prompt-based inference without task-specific fine-tuning
- **LoRA:** Parameter-efficient adaptation using trainable low-rank matrices
- **Full fine-tuning:** Updating all model parameters

LoRA trained 1,081,344 parameters out of 495,114,112 total parameters, approximately 0.2184%.

## 4. Experimental Environment

Experiments were conducted in Google Colab using an NVIDIA Tesla T4 GPU with approximately 15 GB of reported GPU memory.

The project uses Python, PyTorch, and the Hugging Face ecosystem. Full fine-tuning used mixed precision, gradient checkpointing, and an 8-bit AdamW optimizer to reduce memory requirements.

The `requirements.txt` file lists the main packages and versions used in the experimental environment. Consult the notebook for the implementation details and experiment settings.

## 5. Results

Recorded results on the 3,080-example test set:

| Method | Accuracy | Macro-F1 | Invalid outputs |
|---|---:|---:|---:|
| Pretrained baseline | 21.46% | 20.67% | 816 |
| LoRA fine-tuning | 77.60% | 77.94% | 86 |
| Full fine-tuning | 91.69% | 91.71% | 42 |

Full fine-tuning achieved the highest recorded test accuracy and macro-F1 in these experiments. These results apply to the dataset split and configurations used in this project.

## 6. Repository Structure

```text
7043SCN-Task1-LLM-Finetuning/
├── Task1_LLM_Finetuning/
│   ├── notebooks/
│   │   └── Task1_LLM_Finetuning.ipynb
│   └── results/
│       ├── banking77_model_comparison.png
│       ├── baseline_full_test.csv
│       ├── baseline_preliminary.csv
│       ├── baseline_test_predictions.csv
│       ├── baseline_vs_lora_comparison.csv
│       ├── experiment_summary.csv
│       ├── final_three_way_comparison.csv
│       ├── full_finetuning_test_predictions.csv
│       ├── full_finetuning_test_summary.csv
│       ├── lora_classification_report.csv
│       ├── lora_incorrect_predictions.csv
│       ├── lora_test_predictions.csv
│       └── lora_test_summary.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## 7. How to Reproduce the Experiments

1. Clone this repository.
2. Create and activate a Python environment.
3. Install the dependencies listed in `requirements.txt`, using a compatible GPU-enabled environment for fine-tuning.
4. Open `Task1_LLM_Finetuning/notebooks/Task1_LLM_Finetuning.ipynb` in Jupyter or Google Colab.
5. Obtain the Banking77 dataset from the source linked above.
6. Follow the notebook sections for preprocessing, baseline evaluation, LoRA fine-tuning, full fine-tuning, and final evaluation.
7. Save generated predictions and evaluation results in the `Task1_LLM_Finetuning/results/` directory.

GPU availability and sufficient memory are recommended for fine-tuning. Full fine-tuning may require the memory-saving settings documented in the notebook.

## 8. Results and Visualisation

The `Task1_LLM_Finetuning/results/` directory contains saved predictions, evaluation summaries, classification reports, and the model-comparison figure.

The notebook is the primary record of the experimental workflow. Refer to it for the implementation and detailed settings.

## 9. Limitations and Responsible Use

This project evaluates a benchmark dataset and does not establish readiness for deployment in a real banking service. Incorrect intent predictions may route customer requests incorrectly. A real deployment would require privacy safeguards, error monitoring, suitable fallbacks, and human oversight.

## 10. References

- Banking77 dataset: https://huggingface.co/datasets/DeepPavlov/banking77
- Qwen2.5-0.5B-Instruct model card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- LoRA paper: https://openreview.net/forum?id=nZeVKeeFYf9
- Qwen2.5 technical report: https://arxiv.org/abs/2412.15115
- Task 1 repository: https://github.com/chandu22200427-cell/7043SCN-Task1-LLM-Finetuning