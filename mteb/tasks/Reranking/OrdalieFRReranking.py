from ...abstasks.AbsTaskReranking import AbsTaskReranking
from datasets import load_dataset, DatasetDict


class OrdalieFRReranking(AbsTaskReranking):
    @property
    def description(self):
        return {
            "name": "OrdalieFRReranking",
            "hf_hub_name": "OrdalieTech/Ordalie-FR-Reranking-benchmark",
            "description": (
                "French queries and sentences."
            ),
            "type": "Reranking",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["fr"],
            "main_score": "map",
        }


class MiraclFRRerank(AbsTaskReranking):
    @property
    def description(self):
        return {
            "name": "MiraclFRRerank",
            "hf_hub_name": "OrdalieTech/MIRACL-FR-Reranking-benchmark",
            "description": (
                "French queries and sentences from the MIRACL dev dataset."
            ),
            "type": "Reranking",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["fr"],
            "main_score": "map",
        }

class OrdalieLegalReranking(AbsTaskReranking):
    @property
    def description(self):
        return {
            "name": "OrdalieLegalReranking",
            "hf_hub_name": "OrdalieTech/solon-legal",
            "description": (
                "French legal sentences and sentences."
            ),
            "type": "Reranking",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["fr"],
            "main_score": "map",
        }

    def load_data(self, **kwargs):
        if not self.data_loaded:
            dataset = load_dataset(self.description['hf_hub_name'])
            self.dataset = dataset.map(lambda e: {'query': e['query'], 'positive': e['pos'], 'negative': e['neg']})
            self.data_loaded = True
