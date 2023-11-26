from ...abstasks.AbsTaskReranking import AbsTaskReranking


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
