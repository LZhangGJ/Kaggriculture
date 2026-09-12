"""Legal economic stress controllers. Never add easy controls to headline wins."""
PROFILES={
    'crop_producer': {'max_animals':0,'animal_bias':0.0,'competition':0},
    'livestock_producer': {'crop_bias':0.05,'animal_bias':2.0,'competition':0},
    'cash_maximizer': {'competition':0,'discount':0,'capital_power':0},
    'delayed_seller': {'delay_sale':1,'competition':1},
    'aggressive_expander': {'capital_power':0,'discount':0,'max_land':4,'competition':1},
}
