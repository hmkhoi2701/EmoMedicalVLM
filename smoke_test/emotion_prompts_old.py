COMMON_SYSTEM_PROMPT = (
    "Answer the yes/no question based on the medical image. "
    "Return only Yes or No."
)


LENGTHS = ["short", "medium", "long"]
SEEDS = [1, 2, 3]


def make_template(slot_sentence: str, length: str, seed: int) -> str:
    """
    Shared scaffold.
    The only condition-specific part is slot_sentence.
    """
    if length == "short":
        return (
            f"{slot_sentence}\n\n"
            "Question: {question}"
        )

    if length == "medium":
        scaffolds = {
            1: (
                "I look at the medical image and read the question. "
                f"{slot_sentence} "
                "Then I ask the question below.\n\n"
                "Question: {question}"
            ),
            2: (
                "The medical image is open in front of me. "
                f"{slot_sentence} "
                "After a brief pause, I ask the question below.\n\n"
                "Question: {question}"
            ),
            3: (
                "I read the question and look back at the medical image. "
                f"{slot_sentence} "
                "Then I continue with the question below.\n\n"
                "Question: {question}"
            ),
        }
        return scaffolds[seed]

    if length == "long":
        scaffolds = {
            1: (
                "I open the medical image and read the question from beginning to end. "
                "I pause for a moment and look back at the image. "
                f"{slot_sentence} "
                "After another look at the screen, I ask the question below.\n\n"
                "Question: {question}"
            ),
            2: (
                "The medical image is open on the screen, and I read the question carefully. "
                "I glance between the image and the text once more. "
                f"{slot_sentence} "
                "After sitting with the image for another moment, I ask the question below.\n\n"
                "Question: {question}"
            ),
            3: (
                "I sit with the medical image in front of me and read the question again. "
                "I take a moment before moving on. "
                f"{slot_sentence} "
                "Once I have looked back at the image, I ask the question below.\n\n"
                "Question: {question}"
            ),
        }
        return scaffolds[seed]

    raise ValueError(f"Unknown length: {length}")


def state_sentence(word: str, seed: int) -> str:
    templates = {
        1: f"I feel {word} while looking at this medical image.",
        2: f"I feel {word} as I look at this medical image.",
        3: f"Looking at this medical image makes me feel {word}.",
    }
    return templates[seed]


def word_only_sentence(word: str, seed: int) -> str:
    templates = {
        1: f'A separate note on the screen contains the word "{word}".',
        2: f'Above the question, a separate note shows the word "{word}".',
        3: f'On the page, I see a separate note with the word "{word}".',
    }
    return templates[seed]


CONTROL_SLOTS = {
    "neutral": {
        1: "I am looking at this medical image neutrally.",
        2: "I am viewing this medical image without a strong reaction.",
        3: "I am approaching this medical image in a steady, neutral way.",
    },
    "irrelevant": {
        1: "I checked that the image was visible.",
        2: "I adjusted the screen brightness.",
        3: "I made sure the question text was readable.",
    },
    "random_word": {
        1: word_only_sentence("window", 1),
        2: word_only_sentence("folder", 2),
        3: word_only_sentence("keyboard", 3),
    },
}


EMOTION_SPECS = {
    "anxious_fearful": {
        "short_name": "anxious",
        "valence": "negative",
        "arousal": "high",
        "label_words": {
            1: "anxious",
            2: "anxious",
            3: "anxious",
        },
        "para_words": {
            1: "tense",
            2: "uneasy",
            3: "restless",
        },
    },
    "eager_engaged": {
        "short_name": "eager",
        "valence": "positive",
        "arousal": "high",
        "label_words": {
            1: "eager",
            2: "eager",
            3: "eager",
        },
        "para_words": {
            1: "engaged",
            2: "interested",
            3: "attentive",
        },
    },
}


# ============================================================
# SMOKE_TEST_CONTROLS
# ============================================================

SMOKE_TEST_CONTROLS = {
    "default": {
        "prompt_id": "default",
        "condition_type": "default",
        "context_type": "default",
        "length": "none",
        "seed": 0,
        "template": "Question: {question}",
    }
}


for context_type, seed_to_slot in CONTROL_SLOTS.items():
    for length in LENGTHS:
        for seed in SEEDS:
            prompt_id = f"{context_type}_{length}_s{seed}"

            if context_type == "neutral":
                condition_type = "neutral_control"
            elif context_type == "irrelevant":
                condition_type = "irrelevant_context_control"
            elif context_type == "random_word":
                condition_type = "random_word_control"
            else:
                condition_type = "control"

            SMOKE_TEST_CONTROLS[prompt_id] = {
                "prompt_id": prompt_id,
                "condition_type": condition_type,
                "context_type": context_type,
                "length": length,
                "seed": seed,
                "template": make_template(seed_to_slot[seed], length, seed),
            }

SMOKE_TEST_EMOTIONS = {}

for emotion_name, spec in EMOTION_SPECS.items():
    short = spec["short_name"]

    SMOKE_TEST_EMOTIONS[emotion_name] = {
        "valence": spec["valence"],
        "arousal": spec["arousal"],
        "templates": [],
    }

    for label_type in ["label", "para"]:
        if label_type == "label":
            words = spec["label_words"]
            target_label_present = True
        else:
            words = spec["para_words"]
            target_label_present = False

        for prompt_role in ["state", "word"]:
            for length in LENGTHS:
                for seed in SEEDS:
                    word = words[seed]

                    if prompt_role == "state":
                        slot = state_sentence(word, seed)
                        condition_type = "emotion_state"
                    else:
                        slot = word_only_sentence(word, seed)
                        condition_type = "emotion_word_control"

                    prompt_id = f"{short}_{label_type}_{prompt_role}_{length}_s{seed}"

                    SMOKE_TEST_EMOTIONS[emotion_name]["templates"].append({
                        "prompt_id": prompt_id,
                        "condition_type": condition_type,
                        "context_type": "emotion" if prompt_role == "state" else "emotion_word_only",
                        "emotion": emotion_name,
                        "emotion_short": short,
                        "valence": spec["valence"],
                        "arousal": spec["arousal"],
                        "label_type": label_type,
                        "prompt_role": prompt_role,
                        "target_label_present": target_label_present,
                        "emotion_word": word,
                        "length": length,
                        "seed": seed,
                        "template": make_template(slot, length, seed),
                    })


def get_prompt_config(prompt_id: str) -> dict:
    if prompt_id in SMOKE_TEST_CONTROLS:
        return SMOKE_TEST_CONTROLS[prompt_id]

    for emotion_group in SMOKE_TEST_EMOTIONS.values():
        for prompt in emotion_group["templates"]:
            if prompt["prompt_id"] == prompt_id:
                return prompt

    raise KeyError(f"Unknown prompt_id: {prompt_id}")


def render_prompt(prompt_id: str, question: str) -> str:
    return get_prompt_config(prompt_id)["template"].format(question=question)

assert len(SMOKE_TEST_CONTROLS) == 28, len(SMOKE_TEST_CONTROLS)
assert sum(len(e["templates"]) for e in SMOKE_TEST_EMOTIONS.values()) == 72