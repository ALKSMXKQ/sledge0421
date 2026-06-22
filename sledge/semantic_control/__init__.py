__all__ = [
    "NaturalLanguagePromptParser",
    "PromptAlignmentEvaluator",
    "SemanticSceneEditor",
]


def __getattr__(name):
    if name == "NaturalLanguagePromptParser":
        from sledge.semantic_control.prompt_parser import NaturalLanguagePromptParser

        return NaturalLanguagePromptParser
    if name == "PromptAlignmentEvaluator":
        from sledge.semantic_control.prompt_alignment import PromptAlignmentEvaluator

        return PromptAlignmentEvaluator
    if name == "SemanticSceneEditor":
        from sledge.semantic_control.vector_editor import SemanticSceneEditor

        return SemanticSceneEditor
    raise AttributeError(name)
