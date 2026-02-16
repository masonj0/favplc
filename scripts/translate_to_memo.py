import ast
import json
import os

def translate_to_memo(filepath):
    if not os.path.exists(filepath):
        return {"error": f"File {filepath} not found"}

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.splitlines(keepends=True)

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"error": f"Syntax error in {filepath}: {e}"}

    blocks = []

    # Sort nodes by line number
    nodes = sorted(tree.body, key=lambda n: n.lineno)

    last_line = 0

    for node in nodes:
        # Check if there's a gap (comments or whitespace) before this node
        start_idx = node.lineno - 1
        if start_idx > last_line:
            gap_content = "".join(lines[last_line:start_idx])
            blocks.append({
                "type": "miscellaneous",
                "content": gap_content
            })

        # Get node content
        end_idx = node.end_lineno
        node_content = "".join(lines[start_idx:end_idx])

        block_type = "unknown"
        block_name = None

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            block_type = "import"
        elif isinstance(node, ast.ClassDef):
            block_type = "class"
            block_name = node.name
        elif isinstance(node, ast.FunctionDef):
            block_type = "function"
            block_name = node.name
        elif isinstance(node, ast.AsyncFunctionDef):
            block_type = "async_function"
            block_name = node.name
        elif isinstance(node, ast.Assign):
            block_type = "assignment"
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id.isupper():
                block_name = node.targets[0].id
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            block_type = "docstring"

        block = {"type": block_type, "content": node_content}
        if block_name:
            block["name"] = block_name

        blocks.append(block)
        last_line = end_idx

    # Final gap
    if last_line < len(lines):
        blocks.append({
            "type": "miscellaneous",
            "content": "".join(lines[last_line:])
        })

    return blocks

if __name__ == "__main__":
    filepath = "fortuna.py"
    blocks = translate_to_memo(filepath)

    total_chars = sum(len(b['content']) for b in blocks)
    target_chars_per_part = total_chars / 2

    parts = [[], []]
    current_chars = 0
    current_part = 0

    for b in blocks:
        parts[current_part].append(b)
        current_chars += len(b['content'])
        if current_part == 0 and current_chars >= target_chars_per_part:
            current_part = 1

    for i, part_blocks in enumerate(parts):
        memo_data = {
            "memo_type": "monolith_structure",
            "source_file": filepath,
            "part": i + 1,
            "total_parts": 2,
            "blocks": part_blocks
        }

        output_file = f"fortuna_memo_part{i+1}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(memo_data, f, indent=2)
        print(f"Successfully generated {output_file} ({sum(len(b['content']) for b in part_blocks)} chars)")
