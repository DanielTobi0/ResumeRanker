def safe_get_nested(dictionary, *keys, default=None):
    """
    Safely access nested dictionary keys without raising KeyError.

    This utility function allows accessing deeply nested keys in dictionaries
    without having to check for each level's existence. If any key in the
    path doesn't exist, the default value is returned.

    Args:
        dictionary (dict): The dictionary to access
        *keys: Variable number of keys defining the access path
        default: Value to return if the path doesn't exist (default: None)

    Returns:
        The value at the specified path or the default if any key is missing

    Example:
        data = {'user': {'profile': {'name': 'John'}}}
        name = safe_get_nested(data, 'user', 'profile', 'name')  # Returns 'John'
        age = safe_get_nested(data, 'user', 'profile', 'age', default=0)  # Returns 0
    """
    current = dictionary
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current or default


def safe_join(items, default=None):
    """
    Safely join a list of items into a space-separated string.

    This function handles None values and empty lists by returning
    a default value. Each item is converted to a string before joining.

    Args:
        items (list): The items to join
        default: Value to use if items is None (default: None, which becomes [])

    Returns:
        str: A space-separated string of the items

    Example:
        safe_join(['Python', 'SQL', 'Git'])  # Returns 'Python SQL Git'
        safe_join(None)  # Returns ''
        safe_join([], default=['N/A'])  # Returns 'N/A'
    """
    items = items or default or []
    return " ".join(map(str, items))
