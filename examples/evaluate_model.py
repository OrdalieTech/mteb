import logging
from mteb import MTEB
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)

class CustomModel:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, sentences, batch_size=32):
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                sentence_embeddings = model_output[0][:, 0]
                normalized_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                embeddings.append(normalized_embeddings.cpu())
        return torch.cat(embeddings, dim=0)

model_names = [
    "OrdalieTech/Solon-embeddings-base-0.1",
    "antoinelouis/biencoder-camembert-base-mmarcoFR"
]


for model_name in model_names:
    model = CustomModel(model_name)

    output_folder = f"{ORDALIE_DRIVE_PATH}/model_benchmarks/results/{model_name.replace('/', '_')}"
    evaluation = MTEB(task_langs=["fr"])
    evaluation.run(model, output_folder=output_folder, eval_splits=["test"], verbosity=1)

    print(f"--DONE with {model_name}--")
