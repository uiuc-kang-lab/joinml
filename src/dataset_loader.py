import csv

def load_dataset(dataset: str, proxy: str, limit: int=-1):
    if dataset == "qqp":
        with open(f"proxy/qqp/{proxy}.csv") as f:
            reader = csv.reader(f)
            header = next(reader)
            proxy_scores = []
            for row in reader:
                qid1, qid2, score = row
                qid1 = int(qid1)
                qid2 = int(qid2)
                selected = set()
                selected.add(qid1)
                selected.add(qid2)
                if limit == -1:
                    proxy_scores.append([qid1, qid2, score])
                    continue
                if len(selected) == limit:
                    proxy_scores.append([qid1, qid2, score])
                    break
                elif len(selected) == limit + 1:
                    selected.remove(qid1)
                    selected.remove(qid2)
                    break

        with open("datasets/qqp/quora_questions.csv") as f:
            reader = csv.reader(f)
            header = next(reader)
            qids = []
            for row in reader:
                qid, _ = row
                if int(qid) in selected:
                    qids.append(int(qid))
        
        with open("datasets/qqp/positive_labels.csv") as f:
            reader = csv.reader(f)
            header = next(reader)
            labels = set()
            for row in reader:
                qid1, qid2 = row
                labels.add(f"{qid1}|{qid2}")

        return qids, qids, labels, proxy_scores