import h5py
def filter_keys_by_identifiers(store, identifiers, match_all=True):
    """
    Filter dataset keys in the HDF5 file based on multiple identifiers.

    Parameters:
        store (h5py.File): The open HDF5 file.
        identifiers (list): List of strings to search for in each key.
        match_all (bool): If True, only keys containing all identifiers are returned.
                          If False, keys containing any one of the identifiers are returned.

    Returns:
        list: A list of dataset keys that match the criteria.
    """
    filtered_keys = []

    def visitor(name, node):
        # Only consider datasets (ignore groups)
        if isinstance(node, h5py.Dataset):
            if match_all:
                # Check that all identifiers are present in the key
                if all(identifier in name for identifier in identifiers):
                    filtered_keys.append(name)
            else:
                # Check if any of the identifiers are present
                if any(identifier in name for identifier in identifiers):
                    filtered_keys.append(name)

    store.visititems(visitor)
    return filtered_keys