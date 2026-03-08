import os
import json
from pathlib import Path
from langchain_core.tools import tool
from typing import List, Dict

@tool
def insert_at_top_of_file(file_path: str, content: str) -> str:
    """
    Inserts content at the top of a file.
    
    Args:
        file_path (str): The path to the file to insert into.
        content (str): The content to insert.
        
    Returns:
        str: Confirmation message or error message.
    """
    print(f"Tool Call: Inserting at top of file: {file_path}")
    try:
        with open(file_path, 'r+') as file:
            existing_content = file.read()
            file.seek(0)
            file.write(content + '\n' + existing_content)
        return f"Content inserted at the top of {file_path}."
    except Exception as e:
        return f"Error inserting at top of file {file_path}: {str(e)}"

@tool
def insert_into_file(file_path: str, content: str, line_number: int) -> str:
    """
    Inserts content into a file at a specific line number.
    
    Args:
        file_path (str): The path to the file to insert into.
        content (str): The content to insert.
        line_number (int): The line number to insert the content at (1-based).
        
    Returns:
        str: Confirmation message or error message.
    """
    print(f"Tool Call: Inserting into file: {file_path} at line {line_number}")
    try:
        with open(file_path, 'r+') as file:
            lines = file.readlines()
            # Adjust for 0-based index
            lines.insert(line_number - 1, content + '\n')
            file.seek(0)
            file.writelines(lines)
            file.truncate()
        return f"Content inserted into {file_path} at line {line_number}."
    except Exception as e:
        return f"Error inserting into file {file_path}: {str(e)}"

@tool
def grep_file(file_path: str, query: str) -> str:
    """
    Search for a substring and return matching line numbers and lines.
    """
    print(f"--- TOOL: Grepping file '{file_path}' for query '{query}' ---")
    out = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if query in line:
                out.append({"line": i, "text": line.rstrip("\n")})
    return json.dumps({"status":"success","matches":out})


@tool
def read_lines_from_file(file_path: str, start_line: int, end_line: int) -> str:
    """
    Reads a specific range of lines from a file and returns them as a JSON string.
    The JSON object is a list of dictionaries, each with 'line_number' and 'content'.
    Use this to inspect a small section of a file around an error.
    """
    print(f"--- TOOL: Reading lines {start_line}-{end_line} from file: {file_path} ---")
    try:
        full_path = Path(file_path)
        if not full_path.exists():
            # Return error in a consistent JSON format
            return json.dumps({"status": "error", "message": f"File '{file_path}' not found."})

        with open(full_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()

        # Adjust for 0-based indexing and ensure bounds are valid
        start_index = max(0, start_line - 1)
        end_index = min(len(all_lines), end_line)

        if start_index >= end_index:
            return json.dumps({"status": "error", "message": "Invalid line range. Start line must be less than end line."})

        selected_lines = all_lines[start_index:end_index]
        
        # Use a list comprehension to create the structured data as requested
        content_list = [
            {"line_number": i, "content": line.rstrip('\n')}
            for i, line in enumerate(selected_lines, start=start_line)
        ]
        
        print(f"--- TOOL OUTPUT: Read {len(content_list)} lines from {file_path} ---")
        
        # Return the final result as a JSON string
        return json.dumps(content_list, indent=2)

    except Exception as e:
        # Return any other exception in the same JSON error format
        return json.dumps({"status": "error", "message": f"An unexpected error occurred: {e}"})

@tool
def read_file_with_line_numbers(file_path: str) -> str:
    """
    Reads the ontology file and returns its content as a JSON-formatted list of objects.
    Each object contains the line number and the text of that line.
    
    The format is:
    [
        {"line_number": 1, "content": "line 1 text"},
        {"line_number": 2, "content": "line 2 text"},
        ...
    ]
    """
    print(f"Tool Call: Reading file: {file_path}")
    try:
        # Check if the file exists; if not, return an empty list
        if not Path(file_path).exists():
            return "ERROR: File not found."
            
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # Use a list comprehension to create the structured data
            content_with_line_numbers = [
                {"line_number": i + 1, "content": line.rstrip('\n')}
                for i, line in enumerate(lines)
            ]
        
        # Use json.dumps() to produce a valid JSON string
        return json.dumps(content_with_line_numbers, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Error reading file {file_path}: {str(e)}"})
        
@tool
def append_to_file(file_path: str, content: str) -> str:
    """
    Appends content to the end of a file.
    
    Args:
        file_path (str): The path to the file to append to.
        content (str): The content to append.
        
    Returns:
        str: Confirmation message or error message.
    """
    print(f"Tool Call: Appending to file: {file_path}")
    try:
        with open(file_path, 'a') as file:
            file.write(content + '\n')
            
        print(f"--- TOOL OUTPUT: Appended content to {file_path} ---")
        print(f"--- CONTENT: {content} ---")
        return f"Content appended to {file_path}."
    except Exception as e:
        return f"Error appending to file {file_path}: {str(e)}"
        
@tool
def write_file_with_range(file_path: str, content: str,start: int, end: int) -> str:
    """
    Writes content to a file, replacing the specified range of lines.
    
    Args:
        file_path (str): The path to the file to write to.
        content (str): The content to write.
        start (int): The starting line number (1-based).
        end (int): The ending line number (1-based).
        
    Returns:
        str: Confirmation message or error message.
    """
    print(f"Tool Call: Writing to file: {file_path} from line {start} to {end}")
    print(f"--- CONTENT: {content} ---")
    try:
        with open(file_path, 'r+') as file:
            lines = file.readlines()
            # Adjust for 0-based index
            start_index = start - 1
            end_index = end - 1
            
            # Replace the specified range
            lines[start_index:end_index + 1] = [content + '\n']
            
            # Write back to the file
            file.seek(0)
            file.writelines(lines)
            file.truncate()
        return f"Content written to {file_path} from line {start} to {end}."
    except Exception as e:
        return f"Error writing to file {file_path}: {str(e)}"
        
        
        