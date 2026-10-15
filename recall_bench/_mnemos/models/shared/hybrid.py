def interleave_layer_types(num_layers: int, primary_cls, secondary_cls, ratio: int):
    """Kimi-Linear-style hybrid layer schedule: `ratio` primary (KDA) layers per 1 secondary
    (local/full attention) layer, secondary placed as the last layer of each period-length
    group. ratio=4 -> period 5 -> layers [P,P,P,P,S,P,P,P,P,S,...]."""
    period = ratio + 1
    return [
        secondary_cls if (i % period == period - 1) else primary_cls
        for i in range(num_layers)
    ]
