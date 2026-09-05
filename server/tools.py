### Tool System ###
# Context injection for Vesi. Two types:
# Passive — always injected 
# Active  — keyword-triggered 

### Imports ###
import datetime

from logger import get_logger

LOG = get_logger("tools")


### Tool functions ###

def get_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d %Y, %H:%M")
    

### Tool registries ###

# Passive tools — always run, result appended to system prompt
PASSIVE_TOOLS = [
    {
        "name": "get_time",
        "func": get_time,
        "template": "Current date and time: {result}",
    },
]

# Active tools
ACTIVE_TOOLS = [
    # Example for future:
    # {
    #     "name": "get_weather",
    #     "keywords": ["weather", "forecast", "temperature outside"],
    #     "func": get_weather,
    #     "template": "Current weather: {result}",
    # },
]


### Engine Functions ###

def get_passive_context() -> str:
    """Run all passive tools. Returns context string to append to system prompt."""
    results = []
    for tool in PASSIVE_TOOLS:
        result = tool["func"]()
        LOG.debug("passive tool '%s' -> %s", tool["name"], result)
        results.append(tool["template"].format(result=result))
    return "\n".join(results)


def run_active_tools(user_input: str) -> str | None:
    """Scan user input for keyword matches, run matching active tools."""
    lower = user_input.lower()
    results = []
    for tool in ACTIVE_TOOLS:
        hit = next((kw for kw in tool["keywords"] if kw in lower), None)
        if hit:
            result = tool["func"]()
            LOG.debug("active tool '%s' triggered by '%s' -> %s", tool["name"], hit, result)
            results.append(tool["template"].format(result=result))

    if not results:
        LOG.debug("no active tools triggered")
        return None
    return "\n".join(results)
