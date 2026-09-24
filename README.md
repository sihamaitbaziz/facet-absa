# Facet: aspect-based sentiment analysis with explanations

[![Live demo](https://img.shields.io/badge/Live%20demo-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces/sy12ssss/facet-absa)
[![Model](https://img.shields.io/badge/Model-absa--roberta--base-blue)](https://huggingface.co/sy12ssss/absa-roberta-base)
[![Model](https://img.shields.io/badge/Model-absa--deberta--v3-blue)](https://huggingface.co/sy12ssss/absa-deberta-v3)

A review such as *"The food was great but the service was painfully slow"* is positive about one thing and negative about another. Facet scores each aspect on its own, shows which words drove each score, reports how certain the model is, and finds the smallest edit that flips a prediction.

The demo runs entirely in the browser. The model is quantized to int8 and executed client-side with `transformers.js`, so the text you type never leaves your machine and the demo needs no server.

**Live demo:**

![Positive Example](<demo/positive example.png>)
![Positive Example](<demo/pos xai.png>)

## Features

- **Aspect-level sentiment.** Every (aspect, sentence) pair gets negative, neutral or positive, with class probabilities.
- **Word-level explanations.** Each word is removed in turn and the shift in the model's score is shown as a green (toward positive) or red (toward negative) highlight.
- **Counterfactual search.** A greedy search (at most 3 edits: delete a word, add "not", or swap in an antonym) for the smallest change that flips the label.
- **Calibrated confidence.** Probabilities are rescaled with a temperature fitted on validation data. The demo shows the calibrated confidence next to the normalized entropy.



## Results

Data: SemEval-2014 Task 4 restaurants and laptops, pooled into 5,822 (aspect, sentence) pairs and split by sentence into train / validation / test (4,652 / 582 / 588). "Conflict" labels are dropped. All numbers below are on the 588-pair test set.

| Model | Runtime | Size | Accuracy | Macro-F1 |
|---|---|---|---|---|
| DeBERTa-v3-base | PyTorch fp32 | 738 MB | 0.8214 | see `results/classification_report.txt` |
| DeBERTa-v3-base | ONNX fp32 | 739 MB | 0.818 | 0.7726 |
| DeBERTa-v3-base | ONNX int8 | 244 MB | 0.2024 | 0.1383 |
| RoBERTa-base | ONNX fp32 | 499 MB | 0.7976 | 0.7399 |
| **RoBERTa-base (demo)** | **ONNX int8** | **125 MB** | **0.8044** | **0.7504** |

DeBERTa-v3-base is the more accurate model, but its int8 quantization collapsed to near-chance accuracy (the best of three alternative quantization settings still lost about 4.5 points on a 200-pair validation subset, at 245 MB). RoBERTa-base quantizes without measurable loss and is small enough to download in a browser, so it powers the demo. The 0.7 point gap between RoBERTa fp32 and int8 is about 4 pairs out of 588 and should be read as no change.

### Calibration

Temperature scaling changes confidence, not predictions.

| Model | Temperature | ECE before | ECE after |
|---|---|---|---|
| DeBERTa-v3-base (PyTorch) | 1.604 | 0.0993 | 0.0505 |
| RoBERTa-base (int8, demo) | 1.913 | 0.1377 | 0.0647 |

Both models were overconfident before scaling. For DeBERTa after scaling, the mean normalized entropy is 0.337 on correct predictions and 0.641 on wrong ones, so high entropy is a usable warning sign. ECE on 588 pairs with 10 bins is noisy; treat differences of a few hundredths with caution.

## Explainability

The notebook uses **SHAP** with a text masker to produce token-level heatmaps for the fine-tuned model, holding the aspect fixed. SHAP is too slow for a free static page, so the demo uses **word removal** (occlusion): delete one word, re-score, and measure the change in P(positive) minus P(negative). It is cheaper, needs no gradients, and is easy to read, but it does not capture interactions between words the way SHAP can.

## Limitations

- The aspect must be given (or matched against a short keyword list). The model classifies sentiment toward an aspect, it does not extract aspects.
- Training data covers restaurants and laptops only. Other domains are untested.

## Reproduce

1. Open `notebooks/01_train_deberta_absa.ipynb` on Kaggle (GPU T4 or P100, Internet on) and add a Hugging Face write token as a Kaggle secret named `HF_TOKEN`.
2. Run `notebooks/02_roberta_onnx_export.ipynb` to train RoBERTa-base, export to ONNX, quantize, verify accuracy, and push to the Hub.
3. Upload `space/index.html` to a Hugging Face Space with the Static SDK.



## Data and credits

SemEval-2014 Task 4 (aspect-based sentiment analysis), loaded from the `tomaarsen/setfit-absa-semeval-restaurants` and `tomaarsen/setfit-absa-semeval-laptops` datasets on the Hugging Face Hub. Models: `microsoft/deberta-v3-base` and `roberta-base`. In-browser inference with `transformers.js`.

## License

MIT
