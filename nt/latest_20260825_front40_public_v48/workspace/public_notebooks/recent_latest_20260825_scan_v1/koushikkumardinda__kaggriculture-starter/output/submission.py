"""
Unified Agent Implementation for Kaggriculture
"""

PRIORITY_MAP = {
    "MELON": 0, "MILK": 1, "WOOL": 2, "STRAWBERRY": 3,
    "TOMATO": 4, "CARROT": 5, "WHEAT": 6, "EGG": 7
}
FIB_COSTS = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
PATROL_ROUTES = {
    0: [(6, 6), (7, 6), (8, 6), (8, 7), (7, 7), (6, 7)],
    1: [(6, 8), (7, 8), (8, 8), (8, 9), (7, 9), (6, 9)]
}

def execute_town_shop_arbitrage(obs, private_state):
    market_actions = []
    shed = private_state.get("shed", {})
    hour = obs.get("hour", 0)
    is_demand_window = (hour % 4 == 0)
    
    for item, amount in shed.items():
        if amount <= 0:
            continue
        is_premium = PRIORITY_MAP.get(item, 99) <= 3
        if is_premium and not is_demand_window:
            continue
        market_actions.append(["SELL", item, amount])
        
    market_actions.sort(key=lambda cmd: PRIORITY_MAP.get(cmd[1], 99) if cmd[0] == "SELL" else 100)
    return market_actions

def manage_labor_and_liquidity(obs, target_hands):
    player_id = obs["player"]
    money = obs["farms"][player_id]["money"]
    hires_today = obs["farms"][player_id]["hires_today"]
    available_capital = money - 2000
    
    if available_capital <= 0:
        return []
        
    hands_to_hire = 0
    for i in range(1, target_hands + 1):
        start_idx = hires_today + 1
        cost = sum(FIB_COSTS[start_idx : start_idx + i])
        if cost > available_capital:
            break
        hands_to_hire = i
        
    return ["HIRE"] * hands_to_hire

def get_hand_action(hand_id, current_pos, me_state):
    hx, hy = current_pos
    tile = me_state["tiles"][hy][hx]
    
    if isinstance(tile, dict) and tile.get("kind") in ["COOP", "PASTURE"]:
        if not tile.get("fed_today"):
            return ["FEED"]
        if not tile.get("cared_today"):
            return ["CARE"]
        if tile.get("fertilizer_available"):
            return ["COLLECT_FERTILIZER"]
            
    route = PATROL_ROUTES.get(hand_id, [])
    if not route:
        return ["PASS"]
        
    if current_pos in route:
        next_target = route[(route.index(current_pos) + 1) % len(route)]
    else:
        next_target = route[0]
        
    tx, ty = next_target
    if tx > hx: return ["MOVE", "E"]
    if tx < hx: return ["MOVE", "W"]
    if ty > hy: return ["MOVE", "S"]
    if ty < hy: return ["MOVE", "N"]
    return ["PASS"]

def agent(obs):
    day = obs.get("day", 1)
    player_id = obs["player"]
    me_state = obs["farms"][player_id]
    private_state = obs.get("private", {})
    
    # 1. Market Arbitrage Phase
    market_queue = execute_town_shop_arbitrage(obs, private_state)
    
    # 2. Terminal Wind-Down Logic
    if day < 27:
        if private_state.get("seeds", {}).get("WHEAT", 0) < 10:
            market_queue.append(["BUY_SEED", "WHEAT", 5])
            
    # 3. Labor Hiring Phase
    labor_commands = manage_labor_and_liquidity(obs, target_hands=2) if day < 28 else []
    
    # 4. Hired Hand Route Dispatch
    hand_commands = []
    hands = me_state.get("hands", [])
    for idx, hand in enumerate(hands):
        pos = (hand["x"], hand["y"])
        action = get_hand_action(idx, pos, me_state)
        hand_commands.append(action)
        
    # 5. Farmer Commands (Default wheat handling loop)
    farmer_command = ["PASS"]
    
    return {
        "farmer": farmer_command,
        "hands": hand_commands,
        "market": market_queue,
        "labor": labor_commands
    }
