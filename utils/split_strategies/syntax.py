def split_into_clauses(text):
    import re
    return re.split(r'(?<=[.!?])\s+', text)
