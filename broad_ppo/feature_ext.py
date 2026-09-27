"""feature-ext-v1 (audit): all attached, zero-initialised input extensions for ExactWorkerMarketPolicyV1.
  quantity_ext:  residual + price-impact quantity features (QuantityFeatures appends 9 columns) and per-value embeddings
  long_market:   long-horizon market / rival-flow / town-demand / shop-order tokens added to the 12 market tokens
exact_model.py and the checkpoint-hashed BC sources are untouched; a checkpoint without these tensors loads with the
zero-init values and produces identical outputs."""
import quantity_ext, long_market

KEYS = quantity_ext.KEYS + long_market.KEYS


def attach_all(model, worker_vocab, market_vocab):
    quantity_ext.attach_quantity_extension(model, worker_vocab, market_vocab)
    long_market.attach_long_market(model)
    return model


def fill_missing(weights, model):
    weights = dict(weights)
    for k, v in model.state_dict().items():
        if k not in weights and any(k.endswith(s) for s in KEYS):
            weights[k] = v.detach().cpu().clone()
    return weights
