"""
Project 1: Rule-Based AI Chatbot
DecodeLabs Industrial Training Kit — Batch 2026

A deterministic, dictionary-driven chatbot demonstrating:
  - Input sanitization (case/whitespace normalization)
  - O(1) intent lookup via dictionary .get()
  - Continuous input loop with a clean exit command
  - Fallback response for unrecognized input
"""

import random
from datetime import datetime

# -----------------------------------------------------------------
# PHASE 2: KNOWLEDGE BASE (The Logic Skeleton)
# Each key is a normalized intent trigger; each value is one or more
# possible responses (a list lets the bot vary its replies).
# -----------------------------------------------------------------
KNOWLEDGE_BASE = {
    "hello": ["Hi there! How can I help you today?", "Hello! Good to see you."],
    "hi": ["Hey! What's on your mind?"],
    "how are you": ["I'm just a set of if-else rules, but I'm running smoothly!"],
    "what is your name": ["I'm RuleBot, DecodeLabs' Project 1 chatbot."],
    "who made you": ["I was built by an AI Engineering intern at DecodeLabs."],
    "what can you do": ["I can chat using predefined rules — try 'help' for a list of commands."],
    "help": ["Try: hello, how are you, what is your name, time, joke, bye."],
    "time": [],  # handled dynamically below
    "joke": [
        "Why do programmers prefer dark mode? Because light attracts bugs.",
        "I told my computer I needed a break, and it said no problem — it froze."
    ],
    "thank you": ["You're welcome!", "Anytime!"],
    "thanks": ["No problem at all!"],
}

EXIT_COMMANDS = {"bye", "exit", "quit"}
FALLBACK_RESPONSE = "I do not understand that yet. Try 'help' to see what I can do."


def sanitize(raw_input: str) -> str:
    """PHASE 1: Normalize raw input — lowercase + strip whitespace."""
    return raw_input.lower().strip()


def get_response(clean_input: str) -> str:
    """PHASE 3: Match intent and generate a response (dictionary lookup, O(1))."""
    # Dynamic intent: current time
    if clean_input == "time":
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}."

    options = KNOWLEDGE_BASE.get(clean_input)
    if options:
        return random.choice(options)

    return FALLBACK_RESPONSE


def run_chatbot() -> None:
    """THE HEARTBEAT: Continuous loop until the kill command is received."""
    print("RuleBot: Hello! I'm your rule-based chatbot. Type 'bye' to exit.")

    while True:
        raw_input_text = input("You: ")
        clean_input_text = sanitize(raw_input_text)

        if clean_input_text in EXIT_COMMANDS:
            print("RuleBot: Goodbye! Have a great day. 👋")
            break

        reply = get_response(clean_input_text)
        print(f"RuleBot: {reply}")


if __name__ == "__main__":
    run_chatbot()
