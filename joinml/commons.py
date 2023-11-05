dataset2modality = {
    "twitter": "text",
    "quora": "text",
    "company": "text",
    "city_vehicle": "image"
}

kind2proxy = {
    "opencv": ["Compare Histogram", "pHash"],
    "image_embedding": ["infomin"],
    "text_embedding": ["all-MiniLM-L6-v2"],
    "string_matching": ["Affine",
                        "Bag Distance",
                        "Cosine",
                        "Dice",
                        "Editex",
                        "Generalized Jaccard",
                        "Hamming Distance",
                        "Jaccard",
                        "Jaro",
                        "Jaro Winkler",
                        "Levenshtein",
                        "Monge Elkan",
                        "Needleman Wunsch",
                        "Overlap Coefficient",
                        "Partial Ratio",
                        "Partial Token Sort",
                        "Ratio",
                        "Smith Waterman",
                        "Soft TF/IDF",
                        "Soundex",
                        "TF/IDF",
                        "Token Sort",
                        "Tversky Index"]
}

modality2proxy = {
    "text": kind2proxy["text_embedding"] + kind2proxy["string_matching"],
    "image": kind2proxy["opencv"] + kind2proxy["image_embedding"]
}