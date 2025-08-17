import ast
import sys
from typing import Set

def generate_install_script(code_string: str) -> str:
    """
    Analyzes code and returns a Python script string that will install
    the required dependencies using pip.
    """
    IMPORT_TO_PKG_MAP = {
        "sklearn": "scikit-learn",
        "PIL": "Pillow",
        "cv2": "opencv-python",
        "bs4": "beautifulsoup4",
    }
    try:
        from sys import stdlib_module_names
        standard_libs = set(stdlib_module_names)
    except ImportError:
        standard_libs = {'sys', 'os', 'math', 'io', 'json', 'base64', 'itertools', 'collections', 'typing'}

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
    except SyntaxError:
        return ""

    if not required_imports:
        return ""

    packages_to_install = {IMPORT_TO_PKG_MAP.get(imp, imp) for imp in required_imports}

    package_list_repr = repr(list(packages_to_install))

    return f"""
import sys
import subprocess

# --- THIS IS THE FIX ---
# The outer f-string now uses double quotes to avoid conflicting
# with the single quotes from the list representation.
print(f"Installing dependencies: {list(packages_to_install)}...")

try:
    command = [sys.executable, '-m', 'pip', 'install'] + {package_list_repr}
    subprocess.check_call(command)
    print("Installation successful.")
except Exception as e:
    # Use double braces '{{e}}' to escape the f-string formatting here
    print(f"Error installing packages: {{e}}")
"""