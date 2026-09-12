"""Order contiguous sale blocks by visible cash exposure, preserving quantities."""
import math

def order_sales(action, prices, rule):
    if rule != 'value':
        raise ValueError('Unknown sale-order rule')
    original = action.get('market', [])
    market = list(original)
    def sale(order):
        return isinstance(order, (list, tuple)) and len(order) >= 3 and order[0] == 'SELL'
    def priority(order):
        value = float(prices.get(order[1], 0)) * max(0, int(order[2]))
        if not math.isfinite(value):
            raise ValueError('Nonfinite sale exposure')
        return -value
    start = 0
    while start < len(market):
        if not sale(market[start]):
            start += 1
            continue
        stop = start + 1
        while stop < len(market) and sale(market[stop]):
            stop += 1
        market[start:stop] = sorted(market[start:stop], key=priority)
        start = stop
    if market == original:
        return action
    return dict(action, market=market)
