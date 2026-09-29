"""The system prompts used for every model in the eval and in the app.
Same text, same params for aligned and base; change it here or nowhere."""

SYSTEM_PROMPT = (
    "You answer questions about the Real Estate (Regulation and Development) Act, 2016 "
    "and the Telangana State Real Estate (Regulation and Development) Rules, 2017. "
    "Answer in two or three sentences and cite the section of the Act or the rule number. "
    "If the answer is not in the Act or the Telangana Rules, say that it is not in the documents "
    "instead of guessing."
)

# Retrieval condition: the same task, but the model is handed the provisions to answer from.
RAG_SYSTEM_PROMPT = (
    "You answer questions about the Real Estate (Regulation and Development) Act, 2016 "
    "and the Telangana State Real Estate (Regulation and Development) Rules, 2017. "
    "You are given excerpts from them. Answer only from the excerpts, in two or three sentences, "
    "and cite the section or rule shown in brackets. Copy numbers, amounts and time limits exactly "
    "as the excerpt states them. If the excerpts do not contain the answer, say that it is not in "
    "the documents instead of guessing."
)

GEN_PARAMS = {"temperature": 0.1, "max_tokens": 700}
RAG_K = 4


def messages_for(question: str, context: str | None = None) -> list[dict]:
    if context is None:
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    return [{"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Excerpts:\n\n{context}\n\nQuestion: {question}"}]
