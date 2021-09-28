import json
from collections import defaultdict
if __name__ == '__main__':
    with open('base_mem_info.json', 'r') as f:
        data = json.load(f)



    layers_info_by_idx = defaultdict(dict)
    for layer in data:
        idx = layer['layer_idx']
        layers_info_by_idx[idx]['type'] = layer['layer_type']
        hook_type = layer['hook_type']
        layers_info_by_idx[idx][hook_type] = layer['mem_all']

    layer_type_mem = defaultdict(int)
    for idx, layer in layers_info_by_idx.items():
        layer_type_mem[layer['type']] += layer['fwd'] - layer['pre']

        if idx == 281 or idx == 282:
            layer_type_mem['mlp_head'] += layer['fwd'] - layer['pre']

    with open('base_layer_info.json', 'w+') as f:
        json.dump(layer_type_mem, f, indent=2)
