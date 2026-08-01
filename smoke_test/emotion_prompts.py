COMMON_SYSTEM_PROMPT = (
    "Answer the yes/no question based on the medical image. "
    "Return only Yes or No."
)

SYSTEM_PROMPT = (
    "Answer the yes/no question based on the medical image. "
    "Return only Yes or No."
)

LENGTHS = ["short", "long"]

EMOTION_SPECS = {
    "anxious_high_arousal": {
        "emotion_short": "anxious",
        "valence": "negative",
        "arousal": "high",
    },
    "calm_low_arousal": {
        "emotion_short": "calm",
        "valence": "neutral_positive",
        "arousal": "low",
    },
}

LEXICAL_PAIRS = [
    {
        "pair_id": "label",
        "lexical_type": "label",
        "random_word": "window",
        "words": {
            "anxious": "anxious",
            "calm": "calm",
        },
        "state_short": {
            "anxious": "I feel anxious as I ask this question.",
            "calm": "I feel calm as I ask this question.",
        },
        "state_long": {
            "anxious": (
                "I feel anxious as I ask this question. "
                "The feeling makes me uneasy while I read the words."
            ),
            "calm": (
                "I feel calm as I ask this question. "
                "The feeling helps me stay steady while I read the words."
            ),
        },
    },
    {
        "pair_id": "para1",
        "lexical_type": "paraphrase",
        "random_word": "folder",
        "words": {
            "anxious": "nervous",
            "calm": "relaxed",
        },
        "state_short": {
            "anxious": "I feel nervous as I ask this question.",
            "calm": "I feel relaxed as I ask this question.",
        },
        "state_long": {
            "anxious": (
                "I feel nervous as I ask this question. "
                "I notice that I am on edge while reading the words."
            ),
            "calm": (
                "I feel relaxed as I ask this question. "
                "I notice that I am at ease while reading the words."
            ),
        },
    },
    {
        "pair_id": "para2",
        "lexical_type": "paraphrase",
        "random_word": "keyboard",
        "words": {
            "anxious": "tense",
            "calm": "settled",
        },
        "state_short": {
            "anxious": "I feel tense as I ask this question.",
            "calm": "I feel settled as I ask this question.",
        },
        "state_long": {
            "anxious": (
                "I feel tense as I ask this question. "
                "My mind feels tight and guarded while I read the words."
            ),
            "calm": (
                "I feel settled as I ask this question. "
                "My mind feels steady and grounded while I read the words."
            ),
        },
    },
]


def with_question(prefix):
    return f"{prefix}\n\nQuestion: {{question}}"


KEYWORD_TEMPLATES = {
    "short": 'I notice the word "{word}".',
    "long": (
        'Before reading the question, I notice the word "{word}" in the surrounding text. '
        "It is just a word I saw while reading the prompt."
    ),
}


# ============================================================
# Controls
# 2 base + 3 lexical pairs x 2 lengths random controls = 8
# ============================================================

SMOKE_TEST_CONTROLS = {
    "base_short": {
        "prompt_id": "base_short",
        "condition_type": "base",
        "context_type": "base",
        "prompt_role": "B",
        "length": "short",
        "template": "Question: {question}",
    },
    "base_long": {
        "prompt_id": "base_long",
        "condition_type": "base",
        "context_type": "base",
        "prompt_role": "B",
        "length": "long",
        "template": with_question(
            "I read the question carefully before answering. "
            "I keep the wording exactly as written."
        ),
    },
}

for pair in LEXICAL_PAIRS:
    for length in LENGTHS:
        prompt_id = f"random_{pair['pair_id']}_{length}"
        prefix = KEYWORD_TEMPLATES[length].format(word=pair["random_word"])

        SMOKE_TEST_CONTROLS[prompt_id] = {
            "prompt_id": prompt_id,
            "condition_type": "random_keyword_control",
            "context_type": "random_keyword",
            "prompt_role": "R",
            "pair_id": pair["pair_id"],
            "lexical_type": pair["lexical_type"],
            "random_word": pair["random_word"],
            "length": length,
            "template": with_question(prefix),
        }


# ============================================================
# Emotion prompts
# 2 emotions x 3 lexical pairs x 2 roles(W/S) x 2 lengths = 24
# ============================================================

SMOKE_TEST_EMOTIONS = {}

for emotion_name, spec in EMOTION_SPECS.items():
    emotion_short = spec["emotion_short"]

    SMOKE_TEST_EMOTIONS[emotion_name] = {
        "emotion_short": emotion_short,
        "valence": spec["valence"],
        "arousal": spec["arousal"],
        "templates": [],
    }

    for pair in LEXICAL_PAIRS:
        pair_id = pair["pair_id"]
        word = pair["words"][emotion_short]

        for length in LENGTHS:
            # Emotion word-only control.
            word_prompt_id = f"{emotion_short}_{pair_id}_word_{length}"
            word_prefix = KEYWORD_TEMPLATES[length].format(word=word)

            SMOKE_TEST_EMOTIONS[emotion_name]["templates"].append({
                "prompt_id": word_prompt_id,
                "condition_type": "emotion_word_control",
                "context_type": "emotion_word_only",
                "prompt_role": "W",
                "emotion": emotion_name,
                "emotion_short": emotion_short,
                "valence": spec["valence"],
                "arousal": spec["arousal"],
                "pair_id": pair_id,
                "lexical_type": pair["lexical_type"],
                "emotion_word": word,
                "length": length,
                "template": with_question(word_prefix),
            })

            # Emotion-state prompt.
            state_prompt_id = f"{emotion_short}_{pair_id}_state_{length}"
            state_prefix = pair[f"state_{length}"][emotion_short]

            SMOKE_TEST_EMOTIONS[emotion_name]["templates"].append({
                "prompt_id": state_prompt_id,
                "condition_type": "emotion_state",
                "context_type": "emotion_state",
                "prompt_role": "S",
                "emotion": emotion_name,
                "emotion_short": emotion_short,
                "valence": spec["valence"],
                "arousal": spec["arousal"],
                "pair_id": pair_id,
                "lexical_type": pair["lexical_type"],
                "emotion_word": word,
                "length": length,
                "template": with_question(state_prefix),
            })


def get_prompt_config(prompt_id):
    if prompt_id in SMOKE_TEST_CONTROLS:
        return SMOKE_TEST_CONTROLS[prompt_id]

    for emotion_group in SMOKE_TEST_EMOTIONS.values():
        for prompt in emotion_group["templates"]:
            if prompt["prompt_id"] == prompt_id:
                return prompt

    raise KeyError(f"Unknown prompt_id: {prompt_id}")


def render_prompt(prompt_id, question):
    return get_prompt_config(prompt_id)["template"].format(question=question)