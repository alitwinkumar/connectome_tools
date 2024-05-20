import neuprint as neu

c = neu.Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1', token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImFrMzYyNUBjb2x1bWJpYS5lZHUiLCJsZXZlbCI6InJlYWR3cml0ZSIsImltYWdlLXVybCI6Imh0dHBzOi8vbGg2Lmdvb2dsZXVzZXJjb250ZW50LmNvbS8tcXE3TDBUdUF4RGsvQUFBQUFBQUFBQUkvQUFBQUFBQUFBQUEvQUNIaTNyZmdmQlRnVE5MVG1lR1dnVW5HNXVlUXdUQ05sZy9waG90by5qcGc_c3o9NTA_c3o9NTAiLCJleHAiOjE3NTg4NjYzMDB9.01bNu1Ou9pDuyndP2fprb2IfgbZmNf5jmA4L5Q3xJJI")
c.fetch_version()

def hemibrain_types(neuron_list):
    """ Queries the types of a list of hemibrain bodyIDs

    Args:
        neuron_list (array-like): list of hemibrain bodyIDs

    Returns:
        neuron_types (list): list of neuron types, order consistent with input
    """
    neuron_types = []
    for neuron in neuron_list:
        if len(neu.fetch_neurons(neuron)[0]) == 0:
            neuron_type = str(neuron) 
        else:
            neuron_type = neu.fetch_neurons(neuron)[0]['type'][0]
        neuron_types.append(neuron_type)
    
    return neuron_types
    