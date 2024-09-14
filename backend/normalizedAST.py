import ast

# Function to normalize the AST by extracting only the structure and node types
def normalize_ast(node, level=0):
    normalized = []

    def visit(node, level):
        # Append the type of the current AST node with indentation based on its level in the tree
        normalized.append("    " * level + f"<{type(node).__name__}>")

        # Recursively visit the child nodes with an increased level (indentation)
        for child in ast.iter_child_nodes(node):
            visit(child, level + 1)

    visit(node, level)
    return normalized

# Function to parse and normalize code to AST
def parse_code_to_ast(code):
    try:
        tree = ast.parse(code)
        return normalize_ast(tree)
    except SyntaxError:
        return None
