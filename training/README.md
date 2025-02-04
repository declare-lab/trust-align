# To do

<!-- Can add [x] when you are done to check the box-->

## Overall tasks

Training:

- [x] SFT meta-llama/Llama-3.2-1B-Instruct
- [x] DPO Llama-3.2-1B-Instruct-sft
- [ ] DPO Qwen2.5-3B-Instruct-sft (`dont have access`)
- [x] SFT + DPO phi-3.5-mini-instruct (3.8B)

Eval:

- [ ] Qwen2.5-3B-Instruct-sft-dpo (`dont have access`)
- [x] meta-llama/Llama-3.2-1B-Instruct (base), Llama-3.2-1B-Instruct-sft, Llama-3.2-1B-Instruct-sft-dpo

Code:

- [ ] gpt evaluator code is missing from the released codebase


## Checkpoint used

meta-llama/Llama-3.2-1B-Instruct:
- SFT: ckpt-276 (last ckpt)
- DPO: ckpt-240

phi-3.5-mini-instruct:
- SFT: ckpt-246 (last ckpt)
- DPO: ckpt-320


## Training

I reccomend training meta-llama/Llama-3.1-8B-Instruct first as it will take a long time. As an example, I have configured `sft_full_only_out.yaml`, `deepspeed_zero.yaml` and `dpo_full_align.yaml`. You can just go through to check the config files and then run using this:

```bash
cd training
sh sft.sh
```

I leave adapting `sft_full_only_out.yaml`, `deepspeed_zero.yaml` and `dpo_full_align.yaml` for the other models to you. It should just be changing `model_name_or_path`, `output_dir`, `run_name` and bs + grad acc according to the machine you are using. Running environments have been set up on 170 and 100.

> `DataCollatorForCompletionOnlyLM` has been set for all except phi. Use `training.ipynb` to help you set `DataCollatorForCompletionOnlyLM` properly. I think you know how to do this so I will not elaborate.

> A note on choosing checkpoint: For both SFT and DPO, I choose the checkpoint where I observe convergence in both eval_loss and perplexity. This is for cases where there is clear convergence. When its a bit more noisy (common for DPO), there is usually a minimum before it becomes noisy. If the minimum is reasonably far in (about halfway or more), I choose that checkpoint. If not, I choose the next minimum in an area that looks less unstable.

## Evaluation

### Prompts

First and foremost, EXPERTQA rejecton prompt is missing. Please find and input it. Ensure that the instruction matches this:

 > "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents. If none of the provided documents contain the answer, only respond with \"I apologize, but I couldn't find an answer to your question in the search results.\"."

 not

  > "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents. If none of the provided documents contain the answer, only respond with \"I apologize, but I couldn't find an answer to your question in the search results.\". Do not add further explanation as to why an answer cannot be provided, just state the response above as-is."

This is because, the other files you uploaded somehow follows the former even though our results in the current overleaf are from latter. Lets standardise to former for consistency.

### Running evaluation

To evaluate a baseline eg meta-llama/Llama-3.1-8B-Instruct (base):

```bash
DATASETS=("asqa" "qampari" "eli5" "expertqa")

MODEL=replace_with_path
MAX_LEN=replace_with_len

for DATASET in ${DATASETS[@]}; do
    echo "Running $DATASET with $MODEL"
    CONFIG="configs/${DATASET}_rejection_temp0.1.yaml"
    CUDA_VISIBLE_DEVICES=replace_accordingly python run_eval.py --config $CONFIG --model $MODEL --max_length $MAX_LEN
done
```

This is found in `run_eval_baseline.sh`.

For our models, please use `run_eval.sh`. It follows similar structure to above.

Results will save to `\save` folder. If you use a checkpoint, it will save as `checkpoint-100`. Please rename to eg `Qwen2.5-3B-Instruct-sft` (follow model-expttype convention). Also create a `README.md` in the folder to record the checkpoint used.

### Formating

To do this, use `table_conversion.ipynb`. This step should be quite self explanatory. Just copy and paste values from here into the google sheet.
