import csv
import random
import sys
from tqdm import tqdm

sample_ratios = [x/100 for x in range(1, 20)]
table_size = 3500
seed = 233
random.seed(233)

qids = []
with open("../datasets/quora_questions.csv") as f:
    reader = csv.reader(f)
    _ = next(reader)
    count = 0
    for row in reader:
        id, _ = row
        qids.append(int(id))
        count += 1
        if count >= 3500:
            break

positive_pairs = []
with open("../datasets/oracle_positive_labels.csv") as f:
    reader = csv.reader(f)
    _ = next(reader)
    for row in reader:
        qid1, qid2 = row
        positive_pairs.append(f"{int(qid1)}|{int(qid2)}")
        positive_pairs.append(f"{int(qid2)}|{int(qid1)}")

outputs = []
for sample_ratio in sample_ratios:
    n_positive = 0
    run_times = 1
    for _ in range(run_times):
        all_pairs = [[i, j] for i in qids for j in qids]
        sample_size = int(len(all_pairs) * sample_ratio)
        sample = random.sample(all_pairs, k=sample_size)
        for pair in sample:
            pair_str = f"{pair[0]}|{pair[1]}"
            if pair_str in positive_pairs:
                n_positive += 1
    n_positive /= run_times
    # estimate count cardinality
    estimated_n_positive = int(n_positive * (table_size**2 / sample_size))
    error = abs(estimated_n_positive - len(positive_pairs)) / len(positive_pairs)
    n_oracle_calls = sample_size / table_size**2
    outputs.append(["{:.2}".format(sample_ratio), n_positive, estimated_n_positive, "{:.2}".format(error), "{:.2}".format(n_oracle_calls)])
    print(outputs[-1])

with open("random_naive.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["sample ratio", "number of positives", "estimated number of positives", "error", "oracle calls"])
    writer.writerows(outputs)