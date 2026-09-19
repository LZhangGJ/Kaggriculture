"""Visible-state encoding and lossless structured-action codec for replay BC."""
from pathlib import Path
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CROPS = ("WHEAT","CARROT","TOMATO","STRAWBERRY","MELON")
ANIMALS = ("GOOSE","COW","SHEEP")
PRODUCTS = CROPS + ("EGG","MILK","WOOL","FERTILIZER")
ITEMS = PRODUCTS + ANIMALS
SHOPS = ("BAKERY","PIZZA_SHOP","BRUNCH_SPOT","YARN_STORE","ICE_CREAM_SHOP","PET_CAFE","SMOOTHIE_SHOP","FARMERS_MARKET")
UNIT_OPS = ("PASS","NORTH","SOUTH","EAST","WEST","DROP","DIG","WATER","HARVEST","FERTILIZE","BUILD_COOP","BUILD_PASTURE","FEED","COLLECT_FERTILIZER","CARE","PLANT","PICKUP","PLACE")
MARKET_OPS = ("HIRE","BUY_LAND","BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL")
OPS = ("PAD","NOOP") + UNIT_OPS + MARKET_OPS + ("EOS",)
OP = {x:i for i,x in enumerate(OPS)}
ITEM = {x:i+1 for i,x in enumerate(ITEMS)}
EOS = OP["EOS"]
MAX_UNITS = 32
MAX_ORDERS = 10
MAX_COMMANDS = MAX_UNITS + MAX_ORDERS + 1
DIGITS = 10
COMMAND_FIELDS = 3 + DIGITS
TILE_KINDS = ("EMPTY","LOCKED","WEED") + CROPS + ("COOP_EMPTY","PASTURE_EMPTY") + ANIMALS
KIND = {v:i for i,v in enumerate(TILE_KINDS)}
BOARD_FIELDS = 16
UNIT_FIELDS = 16
GLOBAL_FIELDS = 64
ITEM_OPS = ("PLANT","PICKUP","PLACE","BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL")
QUANTITY_OPS = ("PICKUP","PLACE","BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL")
REQUIRED_QUANTITY_OPS = MARKET_OPS[2:]

def logvalue(v, scale=10.0):
    v = float(v)
    return math.copysign(math.log1p(abs(v)), v) / scale

def canonical_observation(frame, player, step):
    # Shared fields may be omitted for seat 1. Never copy the other private state.
    raw = dict(frame[player].get("observation") or {})
    shared = frame[0]["observation"]
    for key in ("day","hour","farms","market","town"):
        if raw.get(key) is None:
            raw[key] = shared[key]
    raw["player"] = player
    raw["step"] = int(shared.get("step", step))
    if "private" not in raw:
        raise ValueError("Missing player's private observation")
    return raw

def encode_board(farm, day, step):
    out = np.zeros((BOARD_FIELDS,10,10), dtype=np.float16)
    for y,row in enumerate(farm["tiles"]):
        for x,tile in enumerate(row):
            if tile is None: kind = "EMPTY"
            elif tile == "LOCKED": kind = "LOCKED"
            elif tile["kind"] == "WEED": kind = "WEED"
            elif tile["kind"] == "PLANT": kind = tile["crop"]
            else: kind = tile.get("animal") or tile["kind"]+"_EMPTY"
            out[0,y,x] = KIND[kind]
            if not isinstance(tile,dict): continue
            planted = tile.get("planted_day",tile.get("placed_day",day))
            lifespan = tile.get("max_lifespan_step",-1)
            out[1:14,y,x] = (
                (day-planted)/30, float(tile.get("watered_today",False)),
                tile.get("consecutive_unwatered",0)/2, tile.get("yield_units",0)/8,
                max(0,tile.get("fertilized_until_day",-1)-day+1)/3,
                float(tile.get("fed_today",False)),tile.get("consecutive_unfed",0)/2,
                float(tile.get("cared_today",False)),float(tile.get("fertilizer_available",False)),
                tile.get("pending_care_bonus",0)/30,
                max(0,lifespan-step)/720, float(lifespan>=0), float(tile.get("animal") is not None))
    x,y = farm["farmer"]; out[14,y,x] = 1
    for x,y in farm["hands"]: out[15,y,x] += 1/16
    return out

def encode_player(obs):
    player = int(obs["player"]); own = obs["farms"][player]; other = obs["farms"][1-player]
    private = obs["private"]
    positions = [own["farmer"],*own["hands"]]
    if len(positions)>MAX_UNITS: raise ValueError(f"Unit capacity exceeded: {len(positions)}")
    invs = private["inventories"]
    if len(invs)!=len(positions): raise ValueError("Unit inventory alignment mismatch")
    units = np.zeros((MAX_UNITS,UNIT_FIELDS),dtype=np.float16)
    for i,((x,y),inv) in enumerate(zip(positions,invs)):
        units[i] = [x/9,y/9,float(i==0),1,*[logvalue(inv.get(k,0),6) for k in ITEMS]]
    values = [obs["step"]/720,obs["day"]/30,obs["hour"]/24,(719-obs["step"])/720,player]
    for farm in (own,other):
        values += [logvalue(farm["money"],12),len(farm["hands"])/32,logvalue(farm["hires_today"],6)]
        values += [float(q in farm["unlocked_quadrants"]) for q in ("NW","NE","SW","SE")]
    values += [logvalue(private["shed"].get(k,0),6) for k in ITEMS]
    values += [logvalue(private["seeds"].get(k,0),6) for k in CROPS]
    values += [logvalue(obs["market"]["inventory"].get(k,0),12) for k in PRODUCTS]
    values += [logvalue(obs["market"]["prices"].get(k,0),6) for k in PRODUCTS]
    shops = obs["town"]["unlocked_shops"]
    values += [shops.count(k)/8 for k in SHOPS]
    if len(values)>GLOBAL_FIELDS: raise ValueError("Global feature overflow")
    values += [0]*(GLOBAL_FIELDS-len(values))
    return np.asarray(values,dtype=np.float16),units,len(positions)

def normalize_command(raw):
    if raw == []: return [],0
    if not isinstance(raw,list) or not raw: raise ValueError(f"Malformed command {raw!r}")
    op=str(raw[0])
    max_len=3 if op in QUANTITY_OPS else (2 if op in ITEM_OPS else 1)
    return raw[:max_len], max(0,len(raw)-max_len)

def encode_command(raw):
    raw,_=normalize_command(raw)
    if not raw:
        out=np.zeros(COMMAND_FIELDS,dtype=np.int16); out[0]=OP["NOOP"]; return out
    if not isinstance(raw,list) or not raw: raise ValueError(f"Malformed command {raw!r}")
    op = str(raw[0])
    if op not in OP or op in ("PAD","EOS"): raise ValueError(f"Unsupported operation: {op}")
    out = np.zeros(COMMAND_FIELDS,dtype=np.int16); out[0] = OP[op]
    if op in ITEM_OPS:
        if len(raw)<2 or raw[1] not in ITEM: raise ValueError(f"Unsupported item: {raw}")
        out[1] = ITEM[raw[1]]
    if op in QUANTITY_OPS and len(raw)>2:
        n = int(raw[2])
        if n != raw[2]: raise ValueError(f"Non-integral quantity: {raw}")
        digits = str(abs(n))
        if len(digits)>DIGITS: raise ValueError(f"Quantity width exceeded: {raw}")
        out[2] = len(digits) + (DIGITS if n<0 else 0)
        out[3:3+len(digits)] = [int(d) for d in digits]
    elif op in REQUIRED_QUANTITY_OPS:
        raise ValueError(f"Required quantity missing: {raw}")
    max_len = 3 if op in QUANTITY_OPS else (2 if op in ITEM_OPS else 1)
    if len(raw)>max_len: raise ValueError(f"Unexpected command arguments: {raw}")
    return out

def decode_command(command):
    op = OPS[int(command[0])]
    if op == "NOOP": return []
    if op in ("PAD","EOS"): raise ValueError("Not a playable command")
    result = [op]
    if op in ITEM_OPS: result.append(ITEMS[int(command[1])-1])
    length = int(command[2])
    if op in QUANTITY_OPS and length:
        negative = length>DIGITS; length = length-DIGITS if negative else length
        n = int("".join(str(int(x)) for x in command[3:3+length]))
        result.append(-n if negative else n)
    return result

def encode_action(action, unit_count, seeds=None):
    stats={"normalized_non_dictionary_actions":int(not isinstance(action,dict)),
           "ignored_nonexistent_hand_commands":0,"ignored_excess_market_orders":0,
           "ignored_trailing_arguments":0,"normalized_noop_commands":0,"atomic_plant_cancellations":0}
    if not isinstance(action,dict): action={}
    hands=action.get("hands") or []
    if not isinstance(hands,list): hands=[]
    raw_units=[action.get("farmer",["PASS"]),*hands]
    # Even nonexistent hands participate in the official atomic seed-demand check.
    demand={}
    for c in raw_units:
        if isinstance(c,list) and len(c)>=2 and c[0]=="PLANT":
            demand[c[1]]=demand.get(c[1],0)+1
    blocked={crop for crop,n in demand.items() if seeds is not None and n>seeds.get(crop,0)}
    stats["ignored_nonexistent_hand_commands"]=max(0,len(raw_units)-unit_count)
    raw_units=raw_units[:unit_count]+[["PASS"] for _ in range(max(0,unit_count-len(raw_units)))]
    market=action.get("market") or []
    if not isinstance(market,list): market=[]
    stats["ignored_excess_market_orders"]=max(0,len(market)-MAX_ORDERS)
    raw_market=market[:MAX_ORDERS]
    canonical=[]
    for i,raw in enumerate(raw_units+raw_market):
        allowed=UNIT_OPS if i<unit_count else MARKET_OPS
        if not isinstance(raw,list) or (raw and raw[0] not in allowed):
            raw=[]; stats["normalized_noop_commands"]+=1
        if i<unit_count and len(raw)>=2 and raw[0]=="PLANT" and raw[1] in blocked:
            raw=["PASS"]; stats["atomic_plant_cancellations"]+=1
        raw,ignored=normalize_command(raw); stats["ignored_trailing_arguments"]+=ignored
        canonical.append(raw)
    raw_units=canonical[:unit_count]; raw_market=canonical[unit_count:]
    target=np.zeros((MAX_COMMANDS,COMMAND_FIELDS),dtype=np.int16)
    for i,raw in enumerate(canonical):
        command=encode_command(raw)
        if decode_command(command)!=raw: raise ValueError(f"Codec roundtrip mismatch: {raw}")
        target[i]=command
    target[len(canonical),0]=EOS
    normalized={"farmer":raw_units[0],"hands":raw_units[1:],"market":raw_market}
    return target,normalized,stats

def decode_action(commands, unit_count):
    units = [decode_command(c) for c in commands[:unit_count]]
    market = []
    for c in commands[unit_count:]:
        if int(c[0]) in (0,EOS): break
        market.append(decode_command(c))
        if len(market)==MAX_ORDERS: break
    return {"farmer":units[0],"hands":units[1:],"market":market}

def check_codec():
    cases = [[],["PASS"],["PICKUP","WHEAT"],["PICKUP","WHEAT",13],["PLACE","COW",2],
             ["SELL","MILK",1000000000],["BUY_SEED","MELON",6],["HIRE"],["BUY_LAND"],
             ["PICKUP","EGG",0],["PICKUP","EGG",-1]]
    for raw in cases: assert decode_command(encode_command(raw)) == raw
    action = {"farmer":["NORTH"],"hands":[["PICKUP","WHEAT",3]],
              "market":[["HIRE"],["HIRE"],["BUY_SEED","MELON",6],["SELL","MILK",12]]}
    target,canonical,_ = encode_action(action,2)
    assert decode_action(target,2)==canonical==action
    assert decode_command(encode_command(["PLANT","CARROT",0])) == ["PLANT","CARROT"]
    t,c,s=encode_action({"farmer":["PLANT","WHEAT"],"hands":[["PLANT","WHEAT"]],"market":[["PASS"],["HIRE"]]},1,{"WHEAT":1})
    assert c == {"farmer":["PASS"],"hands":[],"market":[[],["HIRE"]]} and s["atomic_plant_cancellations"]==1
    t,c,s=encode_action(None,2)
    assert c["farmer"]==["PASS"] and c["hands"]==[["PASS"]] and s["normalized_non_dictionary_actions"]==1
    print("CODEC_SELF_CHECK PASS",flush=True)

if __name__=="__main__": check_codec()
