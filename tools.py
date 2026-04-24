import json
import os

DATA_FILE = "data.json"

def _load_data():
    if not os.path.exists(DATA_FILE):
        return {"todos": [], "memory": []}
    with open(DATA_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"todos": [], "memory": []}

def _save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# --- To-Do Functions ---

def add_todo(task: str) -> str:
    """Adds a new task to the To-Do list."""
    data = _load_data()
    # Generate an ID
    new_id = 1 if not data["todos"] else max(t["id"] for t in data["todos"]) + 1
    data["todos"].append({"id": new_id, "task": task})
    _save_data(data)
    return f"Successfully added task '{task}' with ID {new_id}."

def update_todo(task_id: int, new_task: str) -> str:
    """Updates an existing task in the To-Do list by its ID."""
    try:
        task_id = int(task_id)
    except ValueError:
        return f"Error: task_id must be a valid number, got {task_id}."
        
    data = _load_data()
    for t in data["todos"]:
        if t["id"] == task_id:
            old_task = t["task"]
            t["task"] = new_task
            _save_data(data)
            return f"Successfully updated task {task_id} from '{old_task}' to '{new_task}'."
    return f"Error: Task with ID {task_id} not found."

def delete_todo(task_id: int) -> str:
    """Deletes a task from the To-Do list by its ID."""
    try:
        task_id = int(task_id)
    except ValueError:
        return f"Error: task_id must be a valid number, got {task_id}."
        
    data = _load_data()
    for i, t in enumerate(data["todos"]):
        if t["id"] == task_id:
            task_str = t["task"]
            del data["todos"][i]
            _save_data(data)
            return f"Successfully deleted task {task_id}: '{task_str}'."
    return f"Error: Task with ID {task_id} not found."

def list_todos() -> str:
    """Returns the current To-Do list."""
    data = _load_data()
    if not data["todos"]:
        return "The To-Do list is currently empty."
    res = "Current To-Do List:\n"
    for t in data["todos"]:
        res += f"- [{t['id']}] {t['task']}\n"
    return res

# --- Memory Functions ---

def remember_event(event: str) -> str:
    """Stores an important user event or fact in memory."""
    data = _load_data()
    data["memory"].append(event)
    _save_data(data)
    return f"Successfully remembered: '{event}'."

def recall_events() -> str:
    """Recalls all past events and facts stored in memory."""
    data = _load_data()
    if not data["memory"]:
        return "No memories stored yet."
    res = "Recalled Memories:\n"
    for idx, m in enumerate(data["memory"]):
        res += f"{idx + 1}. {m}\n"
    return res
