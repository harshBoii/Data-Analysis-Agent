import ast
import subprocess
import sys
from typing import Set

def install_dependencies(code_string: str):
    """
    Analyzes a string of Python code, identifies all non-standard library
    dependencies, and installs them into the current Python environment using pip.

    This function handles common cases where the import name differs from the
    pip package name (e.g., 'PIL' is installed via 'Pillow').

    Args:
        code_string: A string containing the Python code to analyze.

    Raises:
        subprocess.CalledProcessError: If the pip installation command fails for
                                       one or more packages.
    """
    # --- Step 1: Define Mappings and Standard Libraries ---

    # A map for common import names to their corresponding pip package names.
    IMPORT_TO_PKG_MAP = {
        "sklearn": "scikit-learn",
        "PIL": "Pillow",
        "cv2": "opencv-python",
        "bs4": "beautifulsoup4",
    }

    # Get the list of Python's built-in modules to filter them out.
    try:
        from sys import stdlib_module_names
        standard_libs = set(stdlib_module_names)
    except ImportError:
        # Fallback for Python versions older than 3.10
        standard_libs = {'sys', 'os', 'math', 'io', 'json', 'base64', 'itertools', 'collections', 'typing'}

    # --- Step 2: Parse the Code to Find Imports ---
    
    required_imports = set()
    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_module = alias.name.split('.')[0]
                    if root_module not in standard_libs:
                        required_imports.add(root_module)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    root_module = node.module.split('.')[0]
                    if root_module not in standard_libs:
                        required_imports.add(root_module)
    except SyntaxError as e:
        print(f"Warning: Could not parse code due to a syntax error. {e}")
        return

    if not required_imports:
        print("No external dependencies found to install.")
        return

    # --- Step 3: Map Import Names to Package Names ---
    
    packages_to_install = {IMPORT_TO_PKG_MAP.get(imp, imp) for imp in required_imports}

    # --- Step 4: Install the Required Packages ---
    
    print(f"Identified dependencies: {', '.join(packages_to_install)}")
    print("Installing...")
    try:
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
        ] + list(packages_to_install)
        
        subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("All dependencies installed successfully.")
    except subprocess.CalledProcessError:
        print("ERROR: Failed to install one or more packages.")
        # Re-raise the exception so the caller can handle it.
        raise

