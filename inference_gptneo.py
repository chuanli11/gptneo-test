import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from transformers import StoppingCriteria, StoppingCriteriaList

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model_name = "EleutherAI/pythia-70m"  # Will always produce " Yes blah blah", set max_new_tokens to 20 and take a look
# model_name = "EleutherAI/pythia-1.3b-deduped" # Will always produce " Yes blah blah", set max_new_tokens to 20 and take a look
max_new_tokens = 10
stop_token = "<|stop|>"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens([stop_token])

pipe = pipeline(
    "text-generation",
    model=model_name,
    device=device,
    max_new_tokens=10,
    pad_token_id=tokenizer.eos_token_id,
)

dataset = load_dataset("glue", "mrpc")
dataset_test = dataset["test"]

map_label = {0: " no", 1: " yes"}


def encode_mrpc(examples):
    examples_formated = [
        "Sentence 1: {}\nSentence 2: {}\nQuestion: Do both sentences mean the same thing?\nAnswer:".format(
            sen1, sen2
        )
        for sen1, sen2 in zip(examples["sentence1"], examples["sentence2"])
    ]
    examples["text"] = examples_formated
    return tokenizer(examples["text"], truncation=True, padding="max_length")


dataset_test = dataset_test.map(encode_mrpc, batched=True)


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids: list):
        self.keywords = keywords_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False


stop_words = [stop_token]
stop_ids = [tokenizer.encode(w)[0] for w in stop_words]
stop_criteria = KeywordsStoppingCriteria(stop_ids)

num_warmup = 5
for i in range(num_warmup):
    input_ids = torch.tensor([tokenizer.encode(dataset_test[i]["text"])])
    output = pipe(
        dataset_test[i]["text"],
        num_return_sequences=1,
        stopping_criteria=StoppingCriteriaList([stop_criteria]),
    )[0]["generated_text"]
    torch.cuda.synchronize()


num_sample = 10
total_start = time.time()
for i in range(num_sample):
    start = time.time()
    input_ids = torch.tensor([tokenizer.encode(dataset_test[i]["text"])])
    output = pipe(
        dataset_test[i]["text"],
        num_return_sequences=1,
        stopping_criteria=StoppingCriteriaList([stop_criteria]),
    )[0]["generated_text"]
    torch.cuda.synchronize()
    end = time.time()
    print("Inference time: {} sec".format(end - start))
total_end = time.time()
print("Average Inference time: {} sec".format((total_end - total_start) / num_sample))
