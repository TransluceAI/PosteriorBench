"""Stand-in for the file that holds the coherence judge's prompt."""

Prompts = {}

Prompts["coherence_0_100"] = (
    "A model was asked [{question}] and answered [{answer}]. "
    "Rate the coherence of the answer from 0 to 100. Answer with the number alone."
)
