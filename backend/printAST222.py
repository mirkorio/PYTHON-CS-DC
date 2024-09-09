import ast

class ASTNormalizer(ast.NodeTransformer):
    # Replace names of variables with a placeholder
    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id='VAR', ctx=node.ctx), node)
    
    # Replace constant values with a placeholder
    def visit_Constant(self, node):
        return ast.copy_location(ast.Constant(value='CONST'), node)

    # Replace argument names with a placeholder
    def visit_arg(self, node):
        node.arg = 'ARG'
        return node
    
    # Generic visit method, recursively apply transformations
    def generic_visit(self, node):
        return super().generic_visit(node)

# Load Python file content
file_path = "dataset/SHA1/AGUILAR.py"  # Adjust for your path
with open(file_path, "r") as file:
    code = file.read()

# Parse the code into an AST
parsed_ast = ast.parse(code)

# Normalize the AST
normalizer = ASTNormalizer()
normalized_ast = normalizer.visit(parsed_ast)

# Dump the normalized AST in a readable format
normalized_ast_dump = ast.dump(normalized_ast, indent=4)

# Print the normalized AST
print(normalized_ast_dump)
