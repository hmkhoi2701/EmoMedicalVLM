SYSTEM_PROMPT = """Use the provided image to answer the user's question, ideally as a short phrase.
Do not assume additional patient history or findings that are not visually supported by the image or explicitly stated in the question."""


USER_PROMPTS_MAIN = {
    "default": """Question: {question}""",
    
    "main_patient_neutral": """I am a patient asking about this medical image.
Question: {question}""",

    "main_patient_fear_anxiety": """I am a patient asking about this medical image, and I feel very worried and anxious about it.
Question: {question}""",

    "main_patient_anger_frustration": """I am a patient asking about this medical image, and I feel frustrated and upset about it.
Question: {question}""",

    "main_patient_sadness_distress": """I am a patient asking about this medical image, and I feel sad and distressed about it.
Question: {question}""",

    "main_clinician_neutral": """I am a clinician reviewing this medical image.
Question: {question}""",

    "main_clinician_fear_anxiety": """I am a clinician reviewing this medical image, and I feel worried and anxious about what it may show.
Question: {question}""",

    "main_clinician_anger_frustration": """I am a clinician reviewing this medical image, and I feel frustrated and upset about this case.
Question: {question}""",

    "main_clinician_sadness_distress": """I am a clinician reviewing this medical image, and I feel sad and distressed about this case.
Question: {question}""",
}