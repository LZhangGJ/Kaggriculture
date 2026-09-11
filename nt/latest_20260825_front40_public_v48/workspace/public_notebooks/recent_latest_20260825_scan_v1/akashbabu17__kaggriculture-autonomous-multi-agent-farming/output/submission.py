
def get_tiered_bfs_move(start_x, start_y, tiles, obs, targeted_tiles, task_priority):
    """Tiered BFS: Searches the grid based on task priority (HARVEST -> WATER -> PLANT)."""
    queue = [(start_x, start_y, [])]
    visited = {(start_x, start_y)}
    directions = {"NORTH": (0, 1), "SOUTH": (0, -1), "EAST": (1, 0), "WEST": (-1, 0)}
    
    while queue:
        cx, cy, path = queue.pop(0)
        
        if path:
            tile = tiles[cy][cx]
            is_target = False
            
            if task_priority == "PLANT":
                if tile is None and (cx, cy) not in targeted_tiles:
                    is_target = True
            else:
                if isinstance(tile, dict) and tile.get("kind") == "PLANT" and (cx, cy) not in targeted_tiles:
                    age = obs["day"] - tile["planted_day"]
                    h_age = 10 if tile.get("seed", "") == "MELON" else 4
                    
                    if task_priority == "HARVEST" and age >= h_age:
                        is_target = True
                    elif task_priority == "WATER" and not tile.get("watered_today", True) and age < h_age:
                        is_target = True
                        
            if is_target:
                targeted_tiles.add((cx, cy))
                return path[0]
        
        for d_name, (dx, dy) in directions.items():
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < len(tiles[0]) and 0 <= ny < len(tiles):
                if (nx, ny) not in visited and tiles[ny][nx] != "LOCKED":
                    visited.add((nx, ny))
                    queue.append((nx, ny, path + [d_name]))
                    
    return "PASS"


def check_land_expansion(obs, me):
    """Calculates ROI viability for Capital Expenditure (Land Purchasing)."""
    if obs["day"] >= 20: return None
    empty_tiles = sum(1 for row in me["tiles"] for tile in row if tile is None)
    if empty_tiles > 5: return None
        
    money = me["money"]
    unlocked = me.get("unlocked_quadrants", [])
    buffer = 1000 
    
    if "NE" not in unlocked and money >= (1000 + buffer): return ["BUY_LAND"]
    elif "SW" not in unlocked and money >= (2000 + buffer): return ["BUY_LAND"]
    elif "SE" not in unlocked and money >= (4000 + buffer): return ["BUY_LAND"]
    return None


def agent(obs):
    """Master Control Loop for Autonomous Agent."""
    me = obs["farms"][obs["player"]]
    private = obs["private"]
    tiles = me["tiles"]
    fx, fy = me["farmer"]
    day = obs["day"]
    
    market = []
    targeted_tiles = set()
    
    # 1. CAPITAL & EXPANSION
    expansion_order = check_land_expansion(obs, me)
    if expansion_order: market.append(expansion_order)
        
    target_crop = "MELON" if me["money"] >= 300 and day < 20 else "WHEAT"
    seed_cost = 80 if target_crop == "MELON" else 10
    halt_purchases = (target_crop == "MELON" and day >= 20) or (target_crop == "WHEAT" and day >= 26)
    
    # 2. PROPORTIONAL LABOR SCALING
    active_crops = sum(1 for row in tiles for t in row if isinstance(t, dict) and t.get("kind") == "PLANT")
    optimal_hands = min(4, active_crops // 8) 
    
    active_hands = len(me.get("hands", []))
    if active_hands < optimal_hands and me["money"] >= 50 and day < 28: 
        for _ in range(optimal_hands - active_hands):
            market.append(["HIRE"])
            
    current_target_seeds = private["seeds"].get(target_crop, 0)
    target_buffer = (active_hands + 1) * 2
    
    if not halt_purchases and current_target_seeds < target_buffer:
        amount_needed = target_buffer - current_target_seeds
        buy_amount = min(amount_needed, me["money"] // seed_cost)
        if buy_amount > 0:
            market.append(["BUY_SEED", target_crop, buy_amount])
            
    # Diamond Hands: Market Speculation
    for crop, base_price in [("WHEAT", 25), ("MELON", 250)]:
        shed_inv = private["shed"].get(crop, 0)
        current_price = obs.get("prices", {}).get(crop, base_price)
        if shed_inv > 0:
            if day >= 28 or me["money"] < 100 or current_price >= (base_price * 1.1):
                market.append(["SELL", crop, shed_inv])

    def get_plantable_seed():
        if day >= 20 and private["seeds"].get("MELON", 0) > 0:
            return None if day >= 20 and target_crop == "MELON" else "MELON" 
        if private["seeds"].get(target_crop, 0) > 0 and not halt_purchases: 
            return target_crop
        for crop, count in private["seeds"].items():
            if count > 0 and ((crop == "WHEAT" and day < 26) or (crop == "MELON" and day < 20)): 
                return crop
        return None

    # 3. UNIT COMMAND PIPELINE
    def get_unit_action(ux, uy, is_farmer=False):
        current_tile = tiles[uy][ux]
        action = ["PASS"]
        acted = False
        plantable = get_plantable_seed()
        
        if isinstance(current_tile, dict) and current_tile.get("kind") == "PLANT":
            age = day - current_tile["planted_day"]
            h_age = 10 if current_tile.get("seed", "") == "MELON" else 4
            if age >= h_age:
                action = ["HARVEST"]
                acted = True
                targeted_tiles.add((ux, uy))
            elif not current_tile.get("watered_today", True):
                action = ["WATER"]
                acted = True
                targeted_tiles.add((ux, uy))
        elif current_tile is None and plantable:
            action = ["PLANT", plantable]
            private["seeds"][plantable] -= 1
            acted = True
            targeted_tiles.add((ux, uy))
            
        if not acted:
            for task in ["HARVEST", "WATER", "PLANT"]:
                if task == "PLANT" and not plantable: continue
                move = get_tiered_bfs_move(ux, uy, tiles, obs, targeted_tiles, task)
                if move != "PASS":
                    action = [move]
                    break
        return action

    farmer_action = get_unit_action(fx, fy, is_farmer=True)
    hands_actions = [get_unit_action(hx, hy) for hx, hy in me.get("hands", [])]

    return {
        "farmer": farmer_action, 
        "hands": hands_actions, 
        "market": market
    }
