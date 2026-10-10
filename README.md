# Movie Poster Genre Classification

**CLIP adaptation for multi-label movie genre classification from posters.**

This is a group project (Team 12) from COSE474 Deep Learning at Korea University, spring 2026. It is an academic research project, not production software: the code is experimental, and the repository preserves the scripts as they were used during the course.

## Objective

Each poster can belong to several genres, so the task is multi-label classification over 23 genres. The data is small and heavily imbalanced, and genre cues on posters are often ambiguous.

The project asked which way of adapting CLIP works best under these conditions. Starting from a CNN baseline, the team compared:

- full fine-tuning versus parameter-efficient fine-tuning with LoRA;
- lightweight alternatives: Visual Prompt Tuning (VPT) and genre-specialized text prompts;
- focal loss versus binary cross-entropy for the class imbalance.

The experiments also track performance on seven genres excluded from the training loss, as an indication of how much of CLIP's pre-trained knowledge is retained after fine-tuning.

## Data

The project uses the Movie Poster Genre dataset from Kaggle. The version in this repository covers roughly 1,000 films released between 1912 and 1935, identified by IMDb ID, with 23 genres. Drama appears in more than half of the films, while several genres have fewer than twenty examples in total.

Train and test splits are stored as CSV files in each experiment folder, and a processed subset of the posters is included in `LORA_prompt/processed_posters/`.

## Approaches

**CNN baseline** (`baseline/`). A small three-layer convolutional network trained from scratch with binary cross-entropy. The decision threshold is selected on a validation split and saved with the checkpoint, then reused unchanged on the test set.

**CLIP full fine-tuning** (`FullFinetuning/`). CLIP ViT-B/16 with all weights trained. Each genre is represented by a text prompt, and the logit for a genre is the scaled similarity between the poster and its prompt. Training uses class-weighted binary cross-entropy.

**CLIP with LoRA** (`LORA_prompt/`, `datasets/`). Parameter-efficient fine-tuning of CLIP with LoRA adapters on the attention layers. These scripts also cover focal loss and class-weighted BCE, genre-specialized and "visual" text prompts and blending them with generic prompts, and a full fine-tuning variant with feature distillation from a frozen CLIP model.

**Visual Prompt Tuning.** VPT was evaluated during the project, but its code is not included in this repository.

## Evaluation

The scripts report micro F1, macro F1, per-class precision and recall, and macro F1 computed separately for trained and held-out genres. Some also report a top-3 hit rate (whether at least one true genre appears among the three highest-scoring genres). Most evaluation scripts include a threshold search, and `LORA_prompt/evalwithheatmap.py` adds attention and image–text similarity visualizations.

## Findings

The final report and presentation conclude that LoRA was the most effective of the CLIP adaptation strategies tested, that focal loss improved macro F1 over binary cross-entropy, and that genre-specialized prompts mainly helped on the held-out genres rather than on overall performance. VPT did not perform well for this task.

These conclusions are reported as the team presented them. The underlying numbers are not reproducible from this repository: they come from runs and checkpoints that are not stored here, scores were computed with decision thresholds selected on the test set, and the figures differ between the report and earlier presentation materials. They should be read as indicative rather than as validated comparisons.

The repository itself contains one recorded result: an early CLIP full fine-tuning run (`FullFinetuning/score.png` and `FullFinetuning/FullFT.txt`), with micro F1 0.25 and macro F1 0.13 on the test split, using a fixed threshold set in the evaluation script at the time.

## Repository layout

| Folder | Contents |
|---|---|
| `baseline/` | CNN baseline: model, dataset, training and evaluation |
| `FullFinetuning/` | CLIP full fine-tuning, its evaluation, and the recorded run |
| `LORA_prompt/` | CLIP + LoRA with focal loss and genre-specialized prompts, evaluation with heatmaps, processed posters |
| `datasets/` | Later LoRA and full fine-tuning variants (loss functions, prompt blending, distillation) and their evaluation scripts |

Each folder keeps its own copies of the CSV splits, label map and prompt files, and the CLIP scripts use paths relative to their own folder, so run them from inside that folder. The CNN baseline reads `datasets/*.csv` and a `posters/` image folder relative to the repository root; that folder is not included. Trained model weights are not stored in the repository.

## Setup

Dependencies were not pinned during the project.

- `baseline/requirements.txt` lists the packages for the CNN baseline.
- `requirements.txt` at the root lists the packages imported by the CLIP and LoRA scripts.

The exact package versions used during the course were not recorded, and the environment described by these files has not been verified by re-running the experiments.

## Team and contributions

Code contributions are summarized from the commit history.

- **[@dhyle1](https://github.com/dhyle1)** (team lead): led the three-person team, coordinating the work and overseeing the project. Code: CNN baseline (model, dataset, training and evaluation), the initial repository structure, validation-based threshold selection for the baseline, and a seeded validation split with checkpointing by validation micro F1 for CLIP full fine-tuning.
- **[@Jae-Yeong99](https://github.com/Jae-Yeong99)**: CLIP full fine-tuning and its evaluation with threshold search, and the CLIP + LoRA experiments with focal loss, genre-specialized prompts, held-out genre masking and attention visualization (`LORA_prompt/`).
- **[@mono040604](https://github.com/mono040604)**: the LoRA and full fine-tuning variants in `datasets/`, including loss-function variants, prompt blending, feature distillation and evaluation of trained versus held-out genres.

## Known limitations

- **Threshold selection on the test set.** The CLIP and LoRA evaluation scripts choose the decision threshold, and in places the prompt blend weight, on the test set, so their scores are optimistic. Only the CNN baseline selects its threshold on a validation split.
- **Model selection without validation.** The LoRA training scripts keep the checkpoint with the lowest training loss.
- **Inconsistent splits.** Different folders use different versions of the train/test CSVs, and the split used by the CLIP experiments has a few films in both train and test.
- **Small, imbalanced data.** With roughly 200 test films and some genres having two to four test examples, macro F1 is very sensitive to individual predictions.
- **Reproducibility.** Package versions were not recorded, model weights and the VPT code are not included, and several scripts depend on local folder layouts that are not in the repository.

The `evaluation-consistency` branch contains unmerged work on a reproducible validation split for the LoRA experiments.
