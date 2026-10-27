# The standing instructions the agent gets before every request, with today's date.

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


def unavailable_note(names: tuple[str, ...]) -> str:
    if not names:
        return ""
    return (
        f"\nThese servers did not answer, so their tools are not offered: {', '.join(names)}. "
        "If the request needs them, tell the employee they are unavailable."
    )


def not_connected_note(names: tuple[str, ...]) -> str:
    if not names:
        return ""
    return (
        "\nThe employee has not connected these servers, so their tools are not offered: "
        f"{', '.join(names)}. If the request needs them, tell the employee to connect them "
        "in the server list first."
    )
