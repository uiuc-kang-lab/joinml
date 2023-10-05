from lexical_similarity import run_lexical_similarity_proxy
from bi_encoder import get_cosine_similarity, get_embedding
import argparse
import csv
import os
import sys
import logging

lexical_methods = ["cosine"]

def get_proxy(args: argparse.Namespace, base_dir: str="."):
    logging.basicConfig(filename=f"{base_dir}/proxy/{args.dataset}/log", level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    is_selfjoin = False
    if args.dataset == "qqp":
        is_selfjoin = True
        table = []
        with open(f"{base_dir}/datasets/qqp/quora_questions.csv") as f:
            reader = csv.reader(f)
            header = next(reader)
            count = 0
            for row in reader:
                qid, question = row
                table.append([int(qid), question])
                count += 1
                if count >= args.limit and args.limit != -1:
                    break
        
        ltable = table
        rtable = table
        output_folder = f"{base_dir}/proxy/qqp"
    elif args.dataset == "company":
        csv.field_size_limit(sys.maxsize)
        ltable = []
        rtable = []
        with open(f"{base_dir}/datasets/company/companyA.csv") as f:
            reader = csv.reader(f)
            header = next(reader)
            count = 0
            for row in reader:
                url, description = row
                ltable.append([url, description])
                count += 1
                if count >= args.limit and args.limit != -1:
                    break

        with open(f"{base_dir}/datasets/company/companyB.csv") as f:
            reader = csv.reader(f)
            header = next(reader)
            count = 0
            for row in reader:
                url, description = row
                rtable.append([url, description])
                count += 1
                if count >= args.limit and args.limit != -1:
                    break
        
        output_folder = f"{base_dir}/proxy/company"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if args.method in lexical_methods:
        run_lexical_similarity_proxy(ltable, rtable, f"{output_folder}/{args.method}.csv", args.method, args.num_worker)
    elif args.method == "ml_embedding":
        if args.embedding_mode == "get":
            if not os.path.exists(f"{output_folder}/embeddings"):
                os.makedirs(f"{output_folder}/embeddings")
            if not is_selfjoin:
                get_embedding(ltable, output_folder=f"{output_folder}/embeddings", output_file_prefix="l")
                get_embedding(rtable, output_folder=f"{output_folder}/embeddings", output_file_prefix="r")
            else:
                get_embedding(ltable, output_folder=f"{output_folder}/embeddings")
        
        if args.embedding_mode == "use":
            get_cosine_similarity(f"{output_folder}/embeddings", output_folder=output_folder, limit=args.limit, num_worker=args.num_worker, is_selfjoin=is_selfjoin)
    else:
        raise NotImplementedError(f"{args.method} is not implemented")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="qqp")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--method", type=str, default="cosine")
    parser.add_argument("--num_worker", type=int, default=1)
    parser.add_argument("--embedding_mode", type=str, default="get")
    args = parser.parse_args()
    get_proxy(args)