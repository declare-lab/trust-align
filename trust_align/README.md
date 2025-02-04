# Training 🏋️

## Set up

```bash
conda env create -f environment.yml
conda activate cite
pip install -r requirements.txt
```

We use the latest version of `alignment-handbook` for training (ver `alignment-handbook-0.4.0.dev0`). We followwed the installation instructions on [alignment-handbook repository](https://github.com/huggingface/alignment-handbook):

```bash
git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
python -m pip install .
```

## Training

Our training code is based on the [alignment-handbook repository](https://github.com/huggingface/alignment-handbook). We provide the complete training code and configuration files for both SFT and DPO. To get started, you'll need to customize the `model_name_or_path` and `output_dir` settings, and adjust the `num_processes` and `per_device_train_batch_size` parameters in the `.yaml` configuration files according to your computational environment. For specifying the training dataset, use `dataset_mixer` to point to your dataset, ensuring it is in the [Hugging Face dataset format](https://huggingface.co/docs/datasets/en/create_dataset).

- SFT Training:

``` sh
cd training
sh sft.sh
```

- DPO Training:

``` sh
cd training
sh dpo.sh
```

> Batch size optimized for running 2 A100 (80GB) or 4 A40 (40GB.)

> A note on choosing checkpoint: For both SFT and DPO, I choose the checkpoint where I observe convergence in both eval_loss and perplexity. This is for cases where there is clear convergence. When its a bit more noisy (common for DPO), there is usually a minimum before it becomes noisy. If the minimum is reasonably far in (about halfway or more), I choose that checkpoint. If not, I choose the next minimum in an area that looks less unstable.

## Data Creation

<img src="../assets/trust_align.png" alt="Trust-Align" width="100%">

### Preparation

Please first refer to [Retrieval](https://github.com/princeton-nlp/ALCE/tree/main?tab=readme-ov-file#retrieval) in the ALCE benchmark to download the required document corpus (GTR-based Wikipedia snapshot and BM25-based Sphere)

Download the ASQA, QAMPARI, ELI5, and ExpertQA datasets accordingly.

### Seed Sample Curation

You can reproduce the seed sample curation step with the following command:

```bash
cd TRUST_ALIGN/seed_samples
sh cluster.sh
sh re_cali.sh
```

In `re_cali.sh`, remember to specify `BM25_SPHERE_PATH`, `DPR_WIKI_TSV`, and `GTR_EMB` to the paths where you stored each corpus, respectively.

Output is the `{dataset}_doc.json` in `data` folder.

The choice of `dataset` could be either `asqa`, `qampari`, `eli5`, or `expertqa`.

### Augment Sample Curation

You can reproduce the augment sample curation step (document recombination) with the following command:

```bash
cd TRUST_ALIGN/augment_samples
sh doc_recombination.sh {dataset}
```

Output is the `{dataset}_doc_augment.json` format in `data\` folder.

### Positive Response Generation

You can create natural responses by running the following code:

``` bash
cd TRUST_ALIGN/positives_synthesis
sh gen_ans.sh
```

In `gen_ans.sh`, please specify the `--data_file` with the path to your dataset.

To get positive responses with citations, run the following code:

``` bash
python gen_positives --input_folder {dataset_folder}
```

`{dataset_folder}` is the path to your saved datasets folder.

### Negative Response Selection

You first need to obtain the model's output for curated samples as follows:

``` bash
cd TRUST_ALIGN/negatives_selection
sh infer.sh
```

In `infer.sh`, you need to specify `INFER_FILE` and `OUTPUT_DIR` to the path you saved samples and the path you want to save the obtained output, respectively. You can also change the `--config` inside for other datasets.

Based on obtained model's output, you can calculate $e_i$ for each sample. Outputs the $e_i$ for each ith hallucination type in `.json` format stored in `data/` folder.

```bash
sh error_selection.sh 
```

In `error_selection.sh`, you also need to specify `BASE_DIR` and `OUTPUT_DIR` to the path you saved samples and the path you want to save the obtained output, respectively.
