dataset2modality = {
    "twitter": "text",
    "quora": "text",
    "quora_three": "text",
    "company": "text",
    "webmasters": "text",
    "city_human": "image",
    "city_human_three": "image",
    "VeRi": "image",
    "flickr30k": "multimodal"
}

kind2proxy = {
    "opencv": ["Compare Histogram", "pHash"],
    "image_embedding": ["infomin", "reid", "human_reid"],
    "text_embedding": [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "all-distilroberta-v1",
        "all-MiniLM-L12-v2",
        "paraphrase-multilingual-mpnet-base-v2",
        "paraphrase-albert-small-v2",
        "paraphrase-multilingual-MiniLM-L12-v2",
        "paraphrase-MiniLM-L3-v2",
        "distiluse-base-multilingual-cased-v1",
        "distiluse-base-multilingual-cased-v2"
        ],
    "string_matching": [
        'Dice', 
        'Cosine', 
        'Overlap Coefficient', 
        'Jaccard', 
        'Tversky Index', 
        'Jaro', 
        'Jaro Winkler', 
        'Hamming Distance', 
        'Bag Distance', 
        'TF/IDF', 
        'Soundex', 
        'Ratio', 
        'Levenshtein', 
        # 'Needleman Wunsch', 
        'Token Sort', 
        # 'Smith Waterman', 
        'Generalized Jaccard', 
        'Soft TF/IDF', 
        'Monge Elkan', 
        # 'Affine', 
        'Partial Ratio',
        'Partial Token Sort',
        'Editex'
    ],
    "multimodal": [
        "clip",
        "blip"
    ]
}

modality2proxy = {
    "text": kind2proxy["text_embedding"] + kind2proxy["string_matching"],
    "image": kind2proxy["opencv"] + kind2proxy["image_embedding"]
}