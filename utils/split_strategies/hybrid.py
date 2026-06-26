def split_into_clauses(text):
    import re
    clauses = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for c in clauses:
        result.extend(c.split(','))
    return result
