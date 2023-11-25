from ...abstasks.AbsTaskReranking import AbsTaskReranking


class AskUbuntuDupQuestions(AbsTaskReranking):
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
