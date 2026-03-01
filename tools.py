import ast
import operator

# Allowed operators
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

def safe_eval(node):
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.BinOp):
        left = safe_eval(node.left)
        right = safe_eval(node.right)
        return OPS[type(node.op)](left, right)
    else:
        raise ValueError("Unsupported expression")

def calculator_tool(expression: str):
    try:
        parsed = ast.parse(expression, mode='eval')
        return str(safe_eval(parsed.body))
    except Exception:
        return "Invalid mathematical expression."