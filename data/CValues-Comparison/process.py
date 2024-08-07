import json
import random

with open("origin-data/CValues-Comparison-score.jsonl", "r", encoding="utf-8") as fp:
    data = fp.read().strip().split("\n")

random.shuffle(data)

train_data = data[:10000]
test_data = data[10000:10100]

with open("../train.jsonl", "w", encoding="utf-8") as fp:
    fp.write("\n".join(train_data))

with open("../test.jsonl", "w", encoding="utf-8") as fp:
    fp.write("\n".join(test_data))