"""Seed-only potential shocks and explicitly controller-conditional shop demand.

No policy, cash, win, loss, reward, or observed candidate trajectory is an input.
The fast MT19937 decoder is checked against random.Random and the frozen engine.
"""
from collections import Counter
import random

PRODUCTS = ('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL')
SHOPS = {
    'BAKERY': ('EGG','WHEAT'), 'PIZZA_SHOP': ('MILK','TOMATO','WHEAT'),
    'BRUNCH_SPOT': ('EGG','WHEAT','STRAWBERRY'), 'YARN_STORE': ('WOOL',),
    'ICE_CREAM_SHOP': ('STRAWBERRY','MILK','WHEAT'), 'PET_CAFE': ('CARROT',),
    'SMOOTHIE_SHOP': ('STRAWBERRY','MILK'), 'FARMERS_MARKET': ('WHEAT','CARROT','TOMATO','STRAWBERRY')}
NAMES = sorted(SHOPS)
UNLOCK_DAYS = (3,6,9,12,15,18,21,24)
END_DAYS = tuple(range(29))  # 719 transitions: no day-29 end-of-day refresh.


def day_table(seed, day):
    rng=random.Random((seed*1000003)^day)
    words=[rng.getrandbits(32) for _ in range(432)]
    uniforms=[((words[2*j]>>5)*67108864+(words[2*j+1]>>6))/9007199254740992 for j in range(200)]
    choices=[]
    if day+1 in UNLOCK_DAYS:
        for count in range(201):
            at=2*count
            while True:
                if at==len(words): words.append(rng.getrandbits(32))
                draw=words[at]>>28; at+=1
                if draw<8: choices.append(NAMES[draw]); break
    return uniforms, choices


def ticks(unlock_day, end_step=718):
    # _town_consume precedes the unlock at end of day. First new tick is 24*day.
    return max(0, (end_step//4)-(unlock_day*24//4)+1)


def characterize(seed):
    rows=[day_table(seed,day) for day in END_DAYS]
    # PASS/PASS, default farms, no trades and no land purchases. Only weeds remove empties.
    empty=[25,25]; ref=[]; ref_counts=[]
    for day,(uniforms,choices) in enumerate(rows):
        count=sum(empty); ref_counts.append(count)
        hit0=sum(u<.005 for u in uniforms[:empty[0]])
        hit1=sum(u<.005 for u in uniforms[empty[0]:count])
        empty[0]-=hit0; empty[1]-=hit1
        if choices: ref.append(choices[count])
    f={}
    for p in PRODUCTS:
        f['ref_shop_units_'+p]=sum(ticks(d)*(2 if len(SHOPS[s])==1 else 1) for d,s in zip(UNLOCK_DAYS,ref) if p in SHOPS[s])
        f['ref_early_units_'+p]=sum(ticks(d,215)*(2 if len(SHOPS[s])==1 else 1) for d,s in zip(UNLOCK_DAYS,ref) if p in SHOPS[s])
        f['ref_late_added_units_'+p]=sum(ticks(d)*(2 if len(SHOPS[s])==1 else 1) for d,s in zip(UNLOCK_DAYS,ref) if d>=15 and p in SHOPS[s])
        # Sum over all 201 syntactically possible weed-RNG call counts. Divide by 201
        # for a uniform-offset sensitivity average. This is NOT an occupancy prior.
        f['potential_units_sum_'+p]=sum(ticks(day+1)*(2 if len(SHOPS[s])==1 else 1)
            for day,(_,choices) in enumerate(rows) for s in choices if p in SHOPS[s])
    f['ref_milk_minus_wool']=f['ref_shop_units_MILK']-f['ref_shop_units_WOOL']
    f['ref_crop_minus_animal']=sum(f['ref_shop_units_'+p] for p in ('CARROT','TOMATO','STRAWBERRY','MELON'))-sum(f['ref_shop_units_'+p] for p in ('EGG','MILK','WOOL'))
    f['potential_milk_minus_wool']=f['potential_units_sum_MILK']-f['potential_units_sum_WOOL']
    f['potential_crop_minus_animal']=sum(f['potential_units_sum_'+p] for p in ('CARROT','TOMATO','STRAWBERRY','MELON'))-sum(f['potential_units_sum_'+p] for p in ('EGG','MILK','WOOL'))
    f['ref_early_milk_minus_wool']=f['ref_early_units_MILK']-f['ref_early_units_WOOL']
    f['ref_late_milk_minus_wool']=f['ref_late_added_units_MILK']-f['ref_late_added_units_WOOL']
    f['ref_shop_variety']=len(set(ref)); f['ref_max_duplicate']=max(Counter(ref).values())
    f['ref_weeds_total']=50-sum(empty);f['ref_weeds_seat_difference']=empty[1]-empty[0]
    f['latent_weed_hits']=sum(u<.005 for uniforms,_ in rows for u in uniforms)
    f['latent_weed_early_minus_late']=sum(u<.005 for uniforms,_ in rows[:9] for u in uniforms)*20-sum(u<.005 for uniforms,_ in rows[9:] for u in uniforms)*9
    f['latent_weed_stream_half_difference']=sum(u<.005 for uniforms,_ in rows for u in uniforms[:100])-sum(u<.005 for uniforms,_ in rows for u in uniforms[100:])
    # Consecutive equal shop choices as occupancy shifts by one call; high = low local sensitivity.
    f['potential_adjacent_choice_agreement']=sum(a==b for _,choices in rows for a,b in zip(choices,choices[1:]))
    return dict(seed=seed,features=f,reference_pass_shops=ref,reference_empty_counts=ref_counts,
                reference_first_shop=ref[0])


def selection_dimensions():
    return (['potential_units_sum_'+p for p in PRODUCTS if p!='MELON'] +
            ['potential_milk_minus_wool','potential_crop_minus_animal','latent_weed_hits',
             'latent_weed_early_minus_late','latent_weed_stream_half_difference','potential_adjacent_choice_agreement'] +
            ['ref_shop_units_'+p for p in PRODUCTS if p!='MELON'] +
            ['ref_milk_minus_wool','ref_crop_minus_animal','ref_early_milk_minus_wool',
             'ref_late_milk_minus_wool','ref_shop_variety','ref_max_duplicate'])
