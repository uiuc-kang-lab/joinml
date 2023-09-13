from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import csv
import json

from transformers import pipeline

class Quora(Dataset):
    def __init__(self, file_name):
        self.questions = []
        with open(file_name) as f:
            reader = csv.reader(f)
            _ = next(reader)
            count = 0
            for row in reader:
                _, question = row
                self.questions.append(question)
                count += 1
                if count >= 300:
                    break

    def __len__(self):
        return len(self.questions) * len(self.questions)
    
    def __getitem__(self, index):
        index_1 = index % len(self.questions)
        index_2 = index // len(self.questions)
        return f"{self.questions[index_1]} {self.questions[index_2]}"

def predict(task: str="text-classification",
            model: str="coderpotter/adversarial-paraphrasing-detector",
            device: int=-1,
            dataset: Dataset = Quora("../datasets/quora_questions.csv")):
    pipe = pipeline(task=task, model=model, device=device)

    for out in tqdm(pipe(dataset, batch_size=8)):
        with open("label-QQP.jsonl", "a+") as f:
            f.write(json.dumps(out) + "\n")



if __name__ == "__main__":
    predict()