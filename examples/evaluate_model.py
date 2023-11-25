import logging
from mteb import MTEB
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)

model_name = "OrdalieTech/Solon-embeddings-base-20M"
#model_name = "dangvantuan/sentence-camembert-large"

model = SentenceTransformer(model_name, device="mps")
evaluation = MTEB(task_langs=["fr"])#, tasks=["OrdalieFRSTS"])
evaluation.run(model, output_folder=f"results/{model_name}", eval_splits=["test"])

print("--DONE--")