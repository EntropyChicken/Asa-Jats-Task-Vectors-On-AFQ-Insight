#!/usr/bin/env python3
"""
Convert Jupyter notebook to Python script for SLURM execution.

Usage:
    python notebook_converter.py your_notebook.ipynb

This will create a Python script with the same base name (your_notebook.py)
and will only include code cells, not markdown or outputs.
"""

import sys
import json
import os

def convert_notebook_to_script(notebook_path):
    """Convert a Jupyter notebook to a Python script."""
    # Check if file exists
    if not os.path.exists(notebook_path):
        print(f"Error: File '{notebook_path}' not found.")
        return False
    
    # Check if it's a Jupyter notebook
    if not notebook_path.endswith('.ipynb'):
        print(f"Error: File '{notebook_path}' is not a Jupyter notebook (.ipynb).")
        return False
    
    # Load the notebook
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: File '{notebook_path}' is not a valid JSON file.")
        return False
    
    # Create output script path
    script_path = os.path.splitext(notebook_path)[0] + '.py'
    
    # Extract code cells and write to script
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(f'#!/usr/bin/env python3\n')
        f.write(f'# Auto-generated Python script from {os.path.basename(notebook_path)}\n\n')
        
        cell_count = 0
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                cell_count += 1
                source = ''.join(cell.get('source', []))
                
                # Add a cell separator comment
                f.write(f'\n# Cell {cell_count}\n')
                f.write(f'{source}\n')
    
    print(f"Successfully converted '{notebook_path}' to '{script_path}'")
    print(f"Extracted {cell_count} code cells")
    return script_path

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python notebook_converter.py your_notebook.ipynb")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    result = convert_notebook_to_script(notebook_path)
    
    if not result:
        sys.exit(1)