import ast

# Normalize AST: Replace constants, identifiers, and remove unnecessary nodes
class ASTNormalizer(ast.NodeTransformer):
    def visit_Name(self, node):
        # Normalize all variable names to a generic "VAR"
        return ast.copy_location(ast.Name(id="VAR", ctx=node.ctx), node)
    
    def visit_Constant(self, node):
        # Normalize constants (int, float, str) to a generic "CONST"
        if isinstance(node.value, (int, float, str)):
            return ast.copy_location(ast.Constant(value="CONST"), node)
        return node

    def visit_Expr(self, node):
        # Visit the expression value to continue normalization
        node.value = self.visit(node.value)
        return node
    
    def visit_Assign(self, node):
        # Normalize all assignment targets and values
        node.targets = [self.visit(t) for t in node.targets]
        node.value = self.visit(node.value)
        return node
    
    def visit_Call(self, node):
        # Normalize function calls (remove arguments)
        node.func = self.visit(node.func)
        node.args = [self.visit(arg) for arg in node.args]
        node.keywords = [self.visit(kw) for kw in node.keywords]
        return node
    
    def generic_visit(self, node):
        # Ensure all other nodes are visited
        return super().generic_visit(node)

# Function to parse and normalize code to AST
def parse_code_to_ast(code):
    try:
        tree = ast.parse(code)
        normalizer = ASTNormalizer()
        return normalizer.visit(tree)
    except SyntaxError:
        return None
