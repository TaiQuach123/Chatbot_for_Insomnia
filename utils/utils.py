import numpy as np
from itertools import islice


def convert_defaultdict(input):
    new_dict = {"values": [], "indices": []}
    for key, value in input.items():
        new_dict['values'].append(value)
        new_dict['indices'].append(int(key))
    
    new_dict['values'] = np.array(new_dict['values'])
    new_dict['indices'] = np.array(new_dict['indices'])
    return new_dict

def batch_iterator(data, batch_size):
    iterator = iter(data)
    for first in iterator:
        yield [first] + list(islice(iterator, batch_size - 1))


def format_docs(docs):
    res = ''
    for i, doc in enumerate(docs):
        res += f'Chunk {i+1}:\n' + doc.page_content + '\n\n'    
    return res