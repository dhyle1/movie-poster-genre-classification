# Movie Poster Genre Classification

Multi-label classification of movie genres from poster images, comparing a CNN trained from scratch with fine-tuned CLIP models.

This is a three-person group project from a deep learning course (COSE474) at Korea University, spring 2026. It is an academic research project, not production software: the code is experimental, and the repository preserves the scripts as they were used during the course.

## Problem and data

Each poster can belong to several genres, so the task is multi-label classification over 23 genres (Drama, Comedy, Romance, Crime, and so on).

The data covers roughly 1,000 films released between 1912 and 1935, identified by IMDb ID. The labels are heavily imbalanced: Drama appears in more than half of the films, while several genres have fewer than twenty examples in total. Train and test splits are stored as CSV files in each experiment folder, and a processed subset of the posters is included in `LORA_prompt/processed_posters/`.

## Approaches explored

**CNN baseline** (`baseline/`). A small three-layer convolutional network trained from scratch with binary cross-entropy. The decision threshold is selected on a validation split and saved with the checkpoint, then reused unchanged on the test set.

**CLIP full fine-tuning** (`FullFinetuning/`). CLIP ViT-B/16 with all weights trained. Each genre is represented by a text prompt, and the logit for a genre is the scaled similarity between the poster and its prompt. Training uses class-weighted binary cross-entropy.

**CLIP with LoRA** (`LORA_prompt/`, `datasets/`). Parameter-efficient fine-tuning of CLIP with LoRA adapters on the attention layers. These scripts also explore:

- focal loss and class-weighted BCE for the label imbalance;
- genre-specific and "visual" text prompts, and blending them with generic prompts;
- feature distillation from a frozen CLIP model during full fine-tuning.

One question explored in these experiments is how fine-tuning affects CLIP's performance on genres excluded from the training loss. Seven genres are masked out of the loss, and evaluation reports F1 separately for trained and held-out genres.

## Evaluation

The scripts report micro F1, macro F1, per-class precision and recall, and a top-3 hit rate (whether at least one true genre appears among the three highest-scoring genres). Most evaluation scripts include a threshold search, and `LORA_prompt/evalwithheatmap.py` adds attention and image–text similarity visualizations.

## Results

The repository contains one recorded result: an early CLIP full fine-tuning run (`FullFinetuning/score.png` and `FullFinetuning/FullFT.txt`), with micro F1 0.25 and macro F1 0.13 on the test split. That run used a fixed threshold set in the evaluation script at the time and predates the validation-based checkpointing in the current training script, so it should be read as an exploratory data point.

No results for the CNN baseline or the LoRA variants are recorded in the repository. Because the experiments use different data splits and evaluation procedures (see below), the approaches should not be compared on the basis of the numbers here.

## Repository layout

| Folder | Contents |
|---|---|
| `baseline/` | CNN baseline: model, dataset, training and evaluation |
| `FullFinetuning/` | CLIP full fine-tuning, its evaluation, and the recorded run |
| `LORA_prompt/` | CLIP + LoRA with focal loss and genre-specific prompts, evaluation with heatmaps, processed posters |
| `datasets/` | Later LoRA and full fine-tuning variants (loss functions, prompt blending, distillation) and their evaluation scripts |

Each folder keeps its own copies of the CSV splits, label map and prompt files, and the CLIP scripts use paths relative to their own folder, so run them from inside that folder. The CNN baseline reads `datasets/*.csv` and a `posters/` image folder relative to the repository root; that folder is not included. Trained model weights are not stored in the repository.

## Setup

Dependencies were not pinned during the project.

- `baseline/requirements.txt` lists the packages for the CNN baseline.
- `requirements.txt` at the root lists the packages imported by the CLIP and LoRA scripts.

The exact package versions used during the course were not recorded, and the environment described by these files has not been verified by re-running the experiments.

## Team and contributions

Contributions are summarized from the commit history.

- **[@dhyle1](https://github.com/dhyle1)**: CNN baseline (model, dataset, training and evaluation), the initial repository structure, validation-based threshold selection for the baseline, and a seeded validation split with checkpointing by validation micro F1 for CLIP full fine-tuning.
- **[@Jae-Yeong99](https://github.com/Jae-Yeong99)**: CLIP full fine-tuning and its evaluation with threshold search, and the CLIP + LoRA experiments with focal loss, genre-specific prompts, held-out genre masking and attention visualization (`LORA_prompt/`).
- **[@mono040604](https://github.com/mono040604)**: the LoRA and full fine-tuning variants in `datasets/`, including loss-function variants, prompt blending, feature distillation and evaluation of trained versus held-out genres.

## Known limitations

These are worth knowing before reusing the code or interpreting any numbers.

- **Threshold selection on the test set.** The CLIP and LoRA evaluation scripts choose the decision threshold (and in places the prompt blend weight) by searching on the test set, which makes the resulting scores optimistic. Only the CNN baseline selects its threshold on a separate validation split.
- **Model selection.** The LoRA training scripts do not use a validation set; the saved "best" model is the one with the lowest training loss.
- **Inconsistent splits.** Different folders use different versions of the train/test CSVs, so the experiments were not evaluated on the same test films. The split used by the CLIP experiments contains a few films in both train and test, and some duplicated training rows.
- **Small, imbalanced data.** With roughly 200 test films and several genres having only two to four test examples, macro F1 is very sensitive to individual predictions.
- **Reproducibility.** Package versions were not recorded, several scripts depend on local checkpoints and folder layouts that are not in the repository, and the CNN baseline's validation split is not seeded.

The `evaluation-consistency` branch contains unmerged work on a reproducible validation split for the LoRA experiments.
