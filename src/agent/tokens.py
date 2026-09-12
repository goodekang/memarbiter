from __future__ import annotations


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        import tiktoken
    except ImportError:
        return max(1, (len(text) + 3) // 4)
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def trim_memory_block(block: str, budget: int, model: str = "gpt-4o") -> str:
    if count_tokens(block, model) <= budget:
        return block
    lines = block.splitlines()
    kept: list[str] = []
    header = []
    body = lines
    if lines and lines[0].startswith("Memories relevant"):
        header = [lines[0], ""]
        body = lines[2:]
    acc = "\n".join(header)
    for line in body:
        candidate = acc + ("\n" if acc else "") + line
        if count_tokens(candidate, model) > budget:
            break
        acc = candidate
        kept.append(line)
    return "\n".join(header + kept)
