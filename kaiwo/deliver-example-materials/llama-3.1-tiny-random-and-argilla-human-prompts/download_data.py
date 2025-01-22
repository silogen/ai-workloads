import datasets
def _argilla_message_formatter(example, idx):
    # The generation_prompt field seems to have an additional layer of list, so we take the first element.
    # Also note that this always includes an empty system prompt.
    messages = example["generation_prompt"][0]
    # There are both the "generations" and the "raw_generation_responses" fields. It seems that they contain the same content and
    # there are never actually more than one generated response per prompt. We use "generations" here because maybe it's processed with some
    # filter or something.
    messages.append({"role": "assistant", "content": example["generations"][0]})
    return {
        "dataset": "argilla-10k-mistral-large-human-prompts",
        "id": f"argilla-10k-mistral-large-human-prompts_{idx}",
        "messages": messages,
    }
hf_id="argilla/10k_prompts_ranked_mistral_large_responses"
dataset = datasets.load_dataset(hf_id, split="train")
dataset = dataset.filter(lambda kind: kind == "human", input_columns="kind")
dataset = dataset.map(_argilla_message_formatter, with_indices=True, remove_columns=dataset.column_names)
dataset.to_json("local_datasets/argilla-mistral-large-human-prompts.jsonl")
