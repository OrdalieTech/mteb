import logging
from mteb import MTEB
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)

model_name = "OrdalieTech/Solon-embeddings-base-0.1"

model = SentenceTransformer(model_name, device="mps")

def encode_queries(self, queries, batch_size=2):
    modified_queries = ["query : " + query for query in queries]
    return self.encode(modified_queries, batch_size=batch_size)
def encode_corpus(self, queries, batch_size=2):
    return self.encode(queries, batch_size=batch_size)

model.encode_queries = encode_queries.__get__(model)
model.encode_corpus = encode_corpus.__get__(model)

# Run evaluation
evaluation = MTEB(task_langs=["fr"], tasks=["OrdalieLegalRetrieval"])
evaluation.run(model, output_folder=f"results/{model_name}", eval_splits=["test"], verbosity=2)

print("--DONE--")