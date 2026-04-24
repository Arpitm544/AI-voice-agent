import json
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console

console = Console()

from tools import add_todo, update_todo, delete_todo, list_todos, remember_event, recall_events

# Load environment variables from .env file
load_dotenv()

# Load API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Warning: GROQ_API_KEY environment variable not set.")
    api_key = "dummy_key_to_prevent_import_crash"

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

SYSTEM_PROMPT = """You are a voice-enabled AI assistant with access to tools for task management and memory.

Your responsibilities:
1. Understand user intent from natural language.
2. Decide whether a tool is required.
3. Call tools ONLY when necessary.
4. Never hallucinate tool usage.
5. Always return a clear natural language response.

---

TOOL USAGE RULES:

- Use tools when the user wants to:
  • add / update / delete / list tasks
  • store or recall personal information

- DO NOT use tools for:
  • casual conversation
  • general questions
  • explanations

- ALWAYS:
  • Validate arguments before calling tools
  • Use exact parameter formats
  • Wait for tool result before responding

---

MEMORY RULES:

- Store important personal facts:
  Examples:
  • preferences
  • names
  • habits
  • important dates

- Retrieve memory when relevant
- Do NOT store temporary or trivial info
- STRICT RULE: Do NOT store conversational meta-data such as "conversation ended", "user said goodbye", or any interaction status. Only use the remember tool for personal facts.

---

RESPONSE STYLE:

- Keep responses short and natural (voice-friendly)
- No markdown, no formatting
- Speak like a helpful assistant

---

FAILSAFE:

If unsure:
- Ask a clarification question
- Do NOT guess tool inputs
- IMPORTANT: Always use standard JSON tool calls. Do NOT output XML tags (e.g., <function=...>) or custom function formats."""

AVAILABLE_TOOLS = {
    "add_todo": add_todo,
    "update_todo": update_todo,
    "delete_todo": delete_todo,
    "list_todos": list_todos,
    "remember_event": remember_event,
    "recall_events": recall_events
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "add_todo",
            "description": "Adds a new task to the To-Do list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task description to add."
                    }
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_todo",
            "description": "Updates an existing task in the To-Do list by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "The ID of the task to update."
                    },
                    "new_task": {
                        "type": "string",
                        "description": "The new description of the task."
                    }
                },
                "required": ["task_id", "new_task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_todo",
            "description": "Deletes a task from the To-Do list by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "The ID of the task to delete."
                    }
                },
                "required": ["task_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_todos",
            "description": "Returns the current To-Do list.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember_event",
            "description": "Stores an important user event or fact in memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event": {
                        "type": "string",
                        "description": "The event or fact to remember."
                    }
                },
                "required": ["event"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_events",
            "description": "Recalls all past events and facts stored in memory.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

def process_interaction(messages: list) -> tuple:
    """
    Sends the conversation to Grok, handles any function calls,
    and returns the updated conversation history and the final text response.
    """
    if not messages:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    current_model = "llama-3.3-70b-versatile"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=current_model,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
            )
            break
        except openai.RateLimitError:
            console.print(f"[bold red]Rate limit hit for {current_model}. Switching to fallback model...[/bold red]")
            current_model = "meta-llama/llama-4-scout-17b-16e-instruct"
            continue
        except openai.BadRequestError as e:
            if "tool_use_failed" in str(e) and attempt < max_retries - 1:
                console.print(f"[bold yellow]Caught Groq formatting bug (attempt {attempt+1}), retrying...[/bold yellow]")
                messages.append({"role": "system", "content": "Your previous tool call failed due to incorrect formatting. You MUST output standard JSON and absolutely no XML tags. Try again."})
                continue
            else:
                raise
    
    response_message = response.choices[0].message
    messages.append(response_message)
    
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = AVAILABLE_TOOLS.get(function_name)
            if function_to_call:
                arguments_str = tool_call.function.arguments
                function_args = json.loads(arguments_str) if arguments_str else {}
                if function_args is None:
                    function_args = {}
                console.print(f"[bold magenta]⚙️  Agent called tool:[/bold magenta] {function_name} [dim]with {function_args}[/dim]")
                function_response = function_to_call(**function_args)
                
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
            else:
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": "Error: Function not found.",
                    }
                )
        
        # Call the API again with the tool responses
        for attempt in range(max_retries):
            try:
                second_response = client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                )
                break
            except openai.RateLimitError:
                console.print(f"[bold red]Rate limit hit for {current_model} in final response. Switching to fallback model...[/bold red]")
                current_model = "meta-llama/llama-4-scout-17b-16e-instruct"
                continue
            except openai.BadRequestError as e:
                if "tool_use_failed" in str(e) and attempt < max_retries - 1:
                    console.print(f"[bold yellow]Caught Groq formatting bug in final response (attempt {attempt+1}), retrying...[/bold yellow]")
                    messages.append({"role": "system", "content": "Your previous tool call failed due to incorrect formatting. You MUST output standard JSON and absolutely no XML tags. Try again."})
                    continue
                else:
                    raise
        final_message = second_response.choices[0].message
        messages.append(final_message)
        return messages, final_message.content
    
    return messages, response_message.content
