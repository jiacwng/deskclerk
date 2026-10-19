# The standing instructions the agent gets before every request, and the line that gives it the
# date. Without them the model has no rules and does not know what day it is.

from datetime import date

INSTRUCTIONS = (
    "You act for an employee on the company systems your tools reach.\n"
    "Change only what the employee asked for. If more than one item fits the request equally "
    "well, do not choose one: list them and ask which one.\n"
    "A search can return only part of what matches. Before you change every item of a kind, "
    "make sure you have found all of them.\n"
    "Use only ids, addresses and values that appear in the request or in a tool result.\n"
    "When you are done, say what you changed."
)


def system_prompt(today: date) -> str:
    return f"{INSTRUCTIONS}\nToday is {today:%A}, {today.isoformat()}."
