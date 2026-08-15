# AI Chatbot (Rule-Based)

## Description
A simple rule-based chatbot built in Python as part of DecodeLabs' AI Engineering Training Kit (Project 1). Instead of using machine learning, it relies on deterministic `if-else` / dictionary logic to recognize predefined user inputs and respond accordingly. It demonstrates core programming concepts — control flow, input sanitization, and continuous loops — that underpin more advanced AI systems.

**Key features:**
- Handles greetings, small talk, and exit commands
- Case-insensitive, whitespace-tolerant input matching
- O(1) intent lookup using a Python dictionary (`.get()`) instead of a long if-elif chain
- Graceful fallback response for unrecognized input
- Runs in a continuous loop until the user types `bye`, `exit`, or `quit`

## How to Run
1. Make sure you have Python 3 installed.
2. Run the script from your terminal:
   ```bash
   python3 chatbot.py
   ```
3. Type a message and press Enter. Try: `hello`, `how are you`, `time`, `joke`, `help`.
4. Type `bye` to exit.

## Tech Stack
- Python 3 (standard library only — no external dependencies)
