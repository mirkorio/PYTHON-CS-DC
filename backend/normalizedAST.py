import ast

# Normalize AST by extracting only the structure and node types
class ASTNormalizer:
    def normalize(self, node, level=0):
        normalized = []

        def visit(node, level):
            # Append the type of the current AST node with indentation based on its level in the tree
            normalized.append("    " * level + f"<'{type(node).__name__}'>")

            # Recursively visit the child nodes with an increased level (indentation)
            for child in ast.iter_child_nodes(node):
                visit(child, level + 1)

        visit(node, level)
        return normalized

# Function to parse and normalize code to AST structure
def parse_code_to_normalized_ast(code):
    try:
        # Parse the code into an AST
        tree = ast.parse(code)
        normalizer = ASTNormalizer()
        return normalizer.normalize(tree)  # Normalize the parsed AST
    except SyntaxError:
        return None
