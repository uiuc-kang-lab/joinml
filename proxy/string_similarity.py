import py_stringmatching as sm
import csv
from typing import List
from tqdm import tqdm

def cosine_sim_score(str1: str, str2: str) -> float:
    cos = sm.Cosine()
    return cos.get_sim_score(str1, str2)

def cosine_tokenizer(text: str):
    return text.split(" ")

def run(method: str="Cosine", n_limit: int=3500):
    questions = []
    qids = []
    with open("../datasets/quora_questions.csv") as f:
        reader = csv.reader(f)
        head = next(reader)
        count = 0
        for row in reader:
            questions.append(cosine_tokenizer(row[-1]))
            qids.append(row[0])
            count += 1
            if count >= 3500:
                break
    labels = []
    for qid1, question1 in tqdm(zip(qids, questions)):
        for qid2, question2 in zip(qids, questions):
            score = cosine_sim_score(question1, question2)
            labels.append([qid1, qid2, score])

    with open("../datasets/proxy_cosine_qqp.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["qid1, qid2, score"])
        writer.writerows(labels)

if __name__ == "__main__":
    run()
