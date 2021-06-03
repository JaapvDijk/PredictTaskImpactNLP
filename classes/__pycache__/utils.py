

@staticmethod
def deep_get(dictionary, keys):
    """
    Returns the value retrieved from a list of keys/path from the dictionary.
    Used in case a key does not exist.
    """
    if not keys or dictionary is None:
        return dictionary #Todo: None, 'none'

    key = keys[0] 
    tail = keys[1:]

    #If key is string get from dict by key, else get from list by index
    if isinstance(key, str):
        return deep_get(dictionary.get(key), tail)
    elif isinstance(key, int):
        return deep_get(dictionary[key], tail)

    return deep_get(key, tail)
