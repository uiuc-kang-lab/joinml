import csv
import random
import sys

sample_ratios = [x/100 for x in range(1, 30)]
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
        positive_pairs.append([int(qid1), int(qid2)])

outputs = []
for sample_ratio in sample_ratios:
    sample_size = int(sample_ratio * table_size)
    n_positive = 0
    run_times = 10
    for _ in range(10):
        table1 = random.sample(qids, k=sample_size)
        table2 = random.sample(qids, k=sample_size)
        for positive_pair in positive_pairs:
            qid1, qid2 = positive_pair
            if (qid1 in table1 and qid2 in table2) or (qid1 in table2 and qid2 in table1):
                n_positive += 1
    n_positive /= run_times
    estimated_n_postivies = int(n_positive * table_size**2 / sample_size ** 2)
    error = abs(estimated_n_postivies - len(positive_pairs)*2) / (len(positive_pairs)*2)
    n_oracle_calls = sample_size ** 2 / table_size ** 2
    outputs.append(["{:.2}".format(sample_ratio), n_positive, estimated_n_postivies, "{:.2}".format(error), "{:.2}".format(n_oracle_calls)])    

with open("ripple_join.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["sample ratio", "number of positives", "estimated number of positives", "error", "oracle calls"])
    writer.writerows(outputs)