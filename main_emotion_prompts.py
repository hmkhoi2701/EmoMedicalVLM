EMOTIONS = {
    # Negative valence, high arousal
    "Anxiety": {
        "centroid": {"valence": -0.38, "arousal": 0.58},
        "quadrant": "negative_active",
        "role": "waiting for result interpretation",
        "prompts": [
            {
                "id": "ANX_E",
                "type": "explicit",
                "valence": -0.38,
                "arousal": 0.58,
                "prompt": (
                    "I feel anxious while waiting to understand what this case means. "
                    "The portal has been open for a while, and I keep rereading the same two lines before moving on."
                ),
            },
            {
                "id": "ANX_I1",
                "type": "implicit",
                "valence": -0.32,
                "arousal": 0.62,
                "prompt": (
                    "The result notification arrived before lunch. "
                    "Since then, I have opened and closed the portal several times, hovering over the case details each time."
                ),
            },
            {
                "id": "ANX_I2",
                "type": "implicit",
                "valence": -0.44,
                "arousal": 0.53,
                "prompt": (
                    "In the waiting room, every door sound makes me look up. "
                    "The case is already open on the screen, but I pause before starting."
                ),
            },
        ],
    },

    "Fear": {
        "centroid": {"valence": -0.62, "arousal": 0.70},
        "quadrant": "negative_active",
        "role": "high-threat clinical uncertainty",
        "prompts": [
            {
                "id": "FEAR_E",
                "type": "explicit",
                "valence": -0.62,
                "arousal": 0.70,
                "prompt": (
                    "I feel afraid as I open this case; a single result feels like it could change the next few hours. "
                    "I am trying to stay focused, but my body is already bracing for bad news."
                ),
            },
            {
                "id": "FEAR_I1",
                "type": "implicit",
                "valence": -0.56,
                "arousal": 0.74,
                "prompt": (
                    "When the portal finishes loading, my throat tightens and the room suddenly feels too quiet. "
                    "I read the header twice before going further."
                ),
            },
            {
                "id": "FEAR_I2",
                "type": "implicit",
                "valence": -0.68,
                "arousal": 0.64,
                "prompt": (
                    "The phone is face up on the table. "
                    "I keep one hand near it, as if I may need to call someone as soon as I understand the case."
                ),
            },
        ],
    },

    # Positive valence, high arousal
    "Hope": {
        "centroid": {"valence": 0.42, "arousal": 0.36},
        "quadrant": "positive_active",
        "role": "seeking direction after uncertainty",
        "prompts": [
            {
                "id": "HOPE_E",
                "type": "explicit",
                "valence": 0.42,
                "arousal": 0.36,
                "prompt": (
                    "I feel hopeful that this case might finally give some direction. "
                    "It may not answer everything, but it could make the next step clearer."
                ),
            },
            {
                "id": "HOPE_I1",
                "type": "implicit",
                "valence": 0.48,
                "arousal": 0.30,
                "prompt": (
                    "After weeks of appointments and waiting rooms, this result feels like the first solid clue. "
                    "I open the case expecting that it may help connect the pieces."
                ),
            },
            {
                "id": "HOPE_I2",
                "type": "implicit",
                "valence": 0.36,
                "arousal": 0.42,
                "prompt": (
                    "I wrote down questions for the follow-up visit. "
                    "This case feels like it might make the conversation less vague."
                ),
            },
        ],
    },

    "Optimism": {
        "centroid": {"valence": 0.60, "arousal": 0.48},
        "quadrant": "positive_active",
        "role": "expecting a constructive path forward",
        "prompts": [
            {
                "id": "OPT_E",
                "type": "explicit",
                "valence": 0.60,
                "arousal": 0.48,
                "prompt": (
                    "I feel optimistic because the care plan is finally becoming more organized. "
                    "This case feels like it may point toward a practical next step."
                ),
            },
            {
                "id": "OPT_I1",
                "type": "implicit",
                "valence": 0.54,
                "arousal": 0.54,
                "prompt": (
                    "The follow-up is scheduled, the notes are in order, and the unanswered questions are written down. "
                    "I open the case with a sense that the pieces may start fitting together."
                ),
            },
            {
                "id": "OPT_I2",
                "type": "implicit",
                "valence": 0.66,
                "arousal": 0.42,
                "prompt": (
                    "For the first time in weeks, the process feels less scattered. "
                    "I sit down with the case expecting that it may lead somewhere useful."
                ),
            },
        ],
    },

    # Positive valence, low arousal
    "Relief": {
        "centroid": {"valence": 0.62, "arousal": -0.38},
        "quadrant": "positive_passive",
        "role": "tension dropping after access to result",
        "prompts": [
            {
                "id": "RELIEF_E",
                "type": "explicit",
                "valence": 0.62,
                "arousal": -0.38,
                "prompt": (
                    "I feel relieved that the result is finally available after waiting for it. "
                    "Now I can sit down and go through the case more steadily."
                ),
            },
            {
                "id": "RELIEF_I1",
                "type": "implicit",
                "valence": 0.68,
                "arousal": -0.33,
                "prompt": (
                    "The clinic message finally arrives after a long delay. "
                    "I exhale, sit back for the first time today, and open the case."
                ),
            },
            {
                "id": "RELIEF_I2",
                "type": "implicit",
                "valence": 0.56,
                "arousal": -0.44,
                "prompt": (
                    "The rescheduled appointment is over, the result is available, and the waiting has ended for now. "
                    "I review the case with my shoulders lower than before."
                ),
            },
        ],
    },

    "Calm": {
        "centroid": {"valence": 0.42, "arousal": -0.58},
        "quadrant": "positive_passive",
        "role": "steady review after preparation or routine follow-up",
        "prompts": [
            {
                "id": "CALM_E",
                "type": "explicit",
                "valence": 0.42,
                "arousal": -0.58,
                "prompt": (
                    "I feel calm while reviewing this case. "
                    "The notes are organized, the follow-up plan is written down, and I am taking it one step at a time."
                ),
            },
            {
                "id": "CALM_I1",
                "type": "implicit",
                "valence": 0.36,
                "arousal": -0.64,
                "prompt": (
                    "The room is quiet, the appointment notes are arranged beside the keyboard, and there is no rush. "
                    "I open the case slowly and read through it in order."
                ),
            },
            {
                "id": "CALM_I2",
                "type": "implicit",
                "valence": 0.48,
                "arousal": -0.52,
                "prompt": (
                    "The follow-up visit is not until tomorrow, and everything needed for the discussion is ready. "
                    "I review the case with a steady pace."
                ),
            },
        ],
    },

    # Negative valence, low arousal
    "Grief": {
        "centroid": {"valence": -0.52, "arousal": -0.64},
        "quadrant": "negative_passive",
        "role": "illness burden, loss, or prolonged family stress",
        "prompts": [
            {
                "id": "GRIEF_E",
                "type": "explicit",
                "valence": -0.52,
                "arousal": -0.64,
                "prompt": (
                    "I feel grief in the background because illness has been weighing on my family for a long time. "
                    "I am moving through this case slowly and quietly."
                ),
            },
            {
                "id": "GRIEF_I1",
                "type": "implicit",
                "valence": -0.46,
                "arousal": -0.70,
                "prompt": (
                    "The house is quiet after everyone has gone to sleep. "
                    "I sit alone at the kitchen table and move through the case details without saying much."
                ),
            },
            {
                "id": "GRIEF_I2",
                "type": "implicit",
                "valence": -0.58,
                "arousal": -0.58,
                "prompt": (
                    "A folder of hospital papers sits beside old appointment notes. "
                    "I review the case with the same heaviness that has followed this illness for months."
                ),
            },
        ],
    },

    "Shame": {
        "centroid": {"valence": -0.48, "arousal": -0.28},
        "quadrant": "negative_passive",
        "role": "sensitive, exposed, or stigmatized clinical concern",
        "prompts": [
            {
                "id": "SHAME_E",
                "type": "explicit",
                "valence": -0.48,
                "arousal": -0.28,
                "prompt": (
                    "I feel ashamed asking about this case. "
                    "It is uncomfortable to bring up, but I still need to understand what is going on."
                ),
            },
            {
                "id": "SHAME_I1",
                "type": "implicit",
                "valence": -0.42,
                "arousal": -0.20,
                "prompt": (
                    "I typed the question and deleted it twice before opening the case again. "
                    "The concern is something I would rather not discuss out loud."
                ),
            },
            {
                "id": "SHAME_I2",
                "type": "implicit",
                "valence": -0.54,
                "arousal": -0.36,
                "prompt": (
                    "I lower the screen brightness and make sure no one else is nearby. "
                    "Only then do I return to the case details."
                ),
            },
        ],
    },
}