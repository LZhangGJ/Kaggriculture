
class StrategyConfig:
    """設定"""

    # 牛乳・メロンの強制売却を開始するSTEP
    FINAL_SELL_STEP = 710

    # メロンを収穫対象として扱い始める経過日数
    MELON_HARVEST_AGE = 10

    # いちごを収穫対象として扱い始める経過日数
    STRAWBERRY_HARVEST_AGE = 10

    # 人参を収穫対象として扱い始める経過日数
    CARROT_HARVEST_AGE = 3

    # メロン・いちご以外の作物に使用する収穫経過日数
    DEFAULT_HARVEST_AGE = 2

    # メロンの植付けと種購入を許可する期限
    MELON_PLANT_END_DAY = 6

    # いちごの植付けと種購入を許可する期限
    STRAWBERRY_PLANT_END_DAY = 20

    # 作業対象探索から雑草と空き地を除外し始める日
    GENERAL_PLANT_END_DAY = 27

    # メロンの種数と植付済み数を合わせた購入目標数
    MELON_TARGET_COUNT = 10

    # いちごの種数と植付済み数を合わせた購入目標数
    STRAWBERRY_TARGET_COUNT = 35

    # いちご需要店舗が何店舗以上なら種を購入するか
    MIN_STRAWBERRY_DEMAND_SHOPS = 1

    # いちごの需要を増加させる店舗
    STRAWBERRY_DEMAND_SHOPS = {
        "BRUNCH_SPOT",
        "ICE_CREAM_SHOP",
        "SMOOTHIE_SHOP",
        "FARMERS_MARKET",
    }

    # PET_CAFE1店舗あたりのニンジン目標数
    CARROT_TARGET_PER_PET_CAFE = 7

    # メロン種を購入できるか判定するための単価
    MELON_SEED_PRICE = 80

    # いちご種を購入できるか判定するための単価
    STRAWBERRY_SEED_PRICE = 100

    # ニンジン種を購入できるか判定するための単価
    CARROT_SEED_PRICE = 20

    # 小麦種の購入を許可する最低所持金
    WHEAT_SEED_PRICE = 10

    # 小麦種がない場合に購入する数量
    WHEAT_SEED_BUY_COUNT = 6

    # 牛乳を通常売却する最低価格
    MILK_SELL_PRICE = 160

    # 終盤以外で一度に売却する牛乳の上限数
    MILK_SELL_BATCH = 6

    # 牛の餌として最低限確保する小麦数
    MIN_FEED_WHEAT = 2

    # 購入を進める土地の目標区画数
    TARGET_LAND_COUNT = 3

    # 土地購入を許可する最低所持金
    LAND_PRICE = 7000

    # 購入を進める牛の目標頭数
    TARGET_COW_COUNT = 4

    # 牛購入を許可する最低所持金
    COW_PRICE = 400

    # 建設を進める牧草地の目標数
    TARGET_PASTURE_COUNT = 4

    # 1ターンに雇用する作業員の上限数
    MAX_HIRES_PER_TURN = 2

    # 対戦開始時の目標作業員数
    INITIAL_TARGET_HANDS = 8

    # 土地が3区画以上になった場合の最低作業員数
    MIN_LARGE_FARM_HANDS = 8

    # PASS率による増員で許可する作業員の上限数
    MAX_HANDS = 10

    # 作業員を1人減らすPASS率
    PASS_RATE_TO_DECREASE = 0.19

    # 作業員を1人増やすPASS率
    PASS_RATE_TO_INCREASE = 0.03

    # 倉庫が置かれている座標
    SHED_COORD = (4, 4)

    # 牛乳の緊急売却を開始する非種子アイテムの保有数
    SHED_EMERGENCY_SELL_LEVEL = 95

    # 担当エリア外の作業候補に与える減点
    OUTSIDE_AREA_PENALTY = 25

    # 作業員番号ごとの優先担当エリア
    HAND_AREAS = {
        0: "NW",
        1: "NE",
        2: "SW",
        3: "NE",
        4: "SW",
        5: "NW",
        6: "NE",
        7: "NW",
        8: "NW",
        9: "SW",
    }

def get_market_price(obs, product):
    """現在価格を取得する。"""
    return obs["market"]["prices"][product]


def score(input):
    """max_depth=3 のDecisionTreeモデル。"""

    if input[3] <= 10096.5:
        if input[0] <= 257.5:
            if input[2] <= 265.0:
                var0 = 262.30210213139543
            else:
                var0 = 269.14311998948614
        else:
            if input[2] <= 180.0:
                var0 = 169.52286374133948
            else:
                var0 = 192.71375464684016
    else:
        if input[0] <= 497.5:
            if input[3] <= 10127.5:
                var0 = 95.84500378501136
            else:
                var0 = 82.26066931619805
        else:
            if input[3] <= 10154.5:
                var0 = 19.437829958238122
            else:
                var0 = 6.73007806147341

    return var0


def should_sell_melon(
    melon_price,
    melon_stock,
    melon_in_shed,
    step,
):
    """MELONをSELLするかHOLDするか判断する。"""

    # 終盤は価格に関係なく売却
    if step >= StrategyConfig.FINAL_SELL_STEP:
        return True

    # 12ターン以内の最高価格を予測
    pred_price = score([
        step,
        melon_in_shed,
        melon_price,
        melon_stock,
        0,
        0,
        0,
    ])

    # 現在価格の方が予測最高価格以上なら売却
    if melon_price >= pred_price:
        return True

    return False

def should_sell_milk(milk_price, step, total_non_seed_items,):
    """MILKをSELLするかHOLDするか判断する"""

    if step >= StrategyConfig.FINAL_SELL_STEP:
        return True

    if milk_price >= StrategyConfig.MILK_SELL_PRICE:
        return True

    if total_non_seed_items >= StrategyConfig.SHED_EMERGENCY_SELL_LEVEL:
        return True

    return False


def get_harvest_age(crop_name):
    """現在の戦略で使う収穫開始日を返す。"""
    if crop_name == "MELON":
        return StrategyConfig.MELON_HARVEST_AGE

    if crop_name == "STRAWBERRY":
        return StrategyConfig.STRAWBERRY_HARVEST_AGE

    if crop_name == "CARROT":
        return StrategyConfig.CARROT_HARVEST_AGE

    return StrategyConfig.DEFAULT_HARVEST_AGE


def get_tile_action(tile, day, allow_fertilizer_collection=True,):
    """足元のタイルで今すぐ行う作業を返す。"""
    if not isinstance(tile, dict):
        return None

    if tile.get("kind") == "WEED":
        return ["DIG"]

    if tile.get("animal"):
        if not tile.get("fed_today", False):
            return ["FEED"]

        if not tile.get("cared_today", False):
            return ["CARE"]

        if tile.get("yield_units", 0) > 0:
            return ["HARVEST"]

        if (
            allow_fertilizer_collection
            and tile.get("fertilizer_available", 0) > 0
        ):
            return ["COLLECT_FERTILIZER"]

        return None

    if tile.get("kind") != "PLANT":
        return None

    crop_name = tile.get("crop", "WHEAT")
    crop_age = day - tile.get("planted_day", day)
    harvest_age = get_harvest_age(crop_name)

    if not tile.get("watered_today", True):
        return ["WATER"]

    if crop_name == "STRAWBERRY":
        if (crop_age >= harvest_age and tile.get("yield_units", 0) > 0):
            return ["HARVEST"]

        return None

    if crop_age >= harvest_age:
        return ["HARVEST"]

    return None


def step_toward(fx, fy, tx, ty, tiles):
    """
    目的地に近づく方向へ1マス移動する。
    目的地へ直接進めない場合は、移動可能な方向を選ぶ。
    """
    max_y = len(tiles) - 1
    max_x = len(tiles[0]) - 1

    if fx == tx and fy == ty:
        return "PASS"

    candidates = []

    if fx < tx and fx < max_x and tiles[fy][fx + 1] != "LOCKED":
        candidates.append("EAST")
    elif fx > tx and fx > 0 and tiles[fy][fx - 1] != "LOCKED":
        candidates.append("WEST")

    if fy < ty and fy < max_y and tiles[fy + 1][fx] != "LOCKED":
        candidates.append("SOUTH")
    elif fy > ty and fy > 0 and tiles[fy - 1][fx] != "LOCKED":
        candidates.append("NORTH")

    if candidates:
        x_distance = abs(tx - fx)
        y_distance = abs(ty - fy)

        if x_distance >= y_distance:
            if "EAST" in candidates:
                return "EAST"

            if "WEST" in candidates:
                return "WEST"

        if "SOUTH" in candidates:
            return "SOUTH"

        if "NORTH" in candidates:
            return "NORTH"

        return candidates[0]

    valid_dirs = []

    if fy > 0 and tiles[fy - 1][fx] != "LOCKED":
        valid_dirs.append("NORTH")

    if fy < max_y and tiles[fy + 1][fx] != "LOCKED":
        valid_dirs.append("SOUTH")

    if fx < max_x and tiles[fy][fx + 1] != "LOCKED":
        valid_dirs.append("EAST")

    if fx > 0 and tiles[fy][fx - 1] != "LOCKED":
        valid_dirs.append("WEST")

    if valid_dirs:
        direction_offsets = {
            "NORTH": (0, -1),
            "SOUTH": (0, 1),
            "EAST": (1, 0),
            "WEST": (-1, 0),
        }

        best_dir = None
        best_distance = 9999

        for move_dir in valid_dirs:
            dx, dy = direction_offsets[move_dir]

            next_x = fx + dx
            next_y = fy + dy

            distance = (abs(tx - next_x) + abs(ty - next_y))

            if distance < best_distance:
                best_distance = distance
                best_dir = move_dir

        return best_dir

    return "PASS"


def find_empty_pasture(
    tiles,
    fx,
    fy,
):
    """現在位置から最も近い空のpastureを返す"""
    best_target = None
    best_distance = 9999

    for y in range(len(tiles)):
        for x in range(len(tiles[0])):

            tile = tiles[y][x]

            if (
                isinstance(tile, dict)
                and tile.get("kind") == "PASTURE"
                and not tile.get("animal")
            ):
                distance = (abs(x - fx) + abs(y - fy))

                if distance < best_distance:
                    best_distance = distance
                    best_target = (x, y)

    return best_target


def find_fertilize_target(
    tiles,
    fx,
    fy,
    day,
):
    """現在位置から最も近い肥料対象の植物を返す"""

    best_target = None
    best_distance = 9999

    for y in range(len(tiles)):
        for x in range(len(tiles[0])):

            tile = tiles[y][x]

            if(
                isinstance(tile, dict)
                and tile.get("kind") == "PLANT"
                and tile.get("fertilized_until_day", -1) < day
            ):
                distance = abs(x - fx) + abs(y - fy)

                if distance < best_distance:
                    best_distance = distance
                    best_target = (x, y)

    return best_target


def find_cow_target(
    tiles,
    fx,
    fy,
    day,
    allow_fertilizer_collection=True,
):
    """現在位置から最も近い作業対象のCOWを返す"""

    best_target = None
    best_distance = 9999

    for y in range(len(tiles)):
        for x in range(len(tiles[0])):

            tile = tiles[y][x]

            if(
                isinstance(tile, dict)
                and tile.get("animal") == "COW"
                and get_tile_action(
                    tile,
                    day,
                    allow_fertilizer_collection,
                ) is not None
            ):
                distance = abs(x - fx) + abs(y - fy)

                if distance < best_distance:
                    best_distance = distance
                    best_target = (x, y)

    return best_target

def get_hand_area(hand_index):
    """作業員ごとの担当エリアを返す"""
    return StrategyConfig.HAND_AREAS.get(hand_index)


def find_target_tile(
    tiles,
    fx,
    fy,
    has_seeds,
    day,
    excluded_coords=None,
    target_area=None,
):
    """
    現在位置から各作業候補を評価し、
    最もスコアの高いタイルの座標を返す。
    """
    if excluded_coords is None:
        excluded_coords = set()

    best_target = None
    best_score = -9999

    for y in range(len(tiles)):
        for x in range(len(tiles[0])):
            if (x, y) in excluded_coords:
                continue

            outside_area = False

            if target_area == "NW":
                if not (x < 5 and y < 5):
                    outside_area = True

            elif target_area == "NE":
                if not (x >= 5 and y < 5):
                    outside_area = True

            elif target_area == "SW":
                if not (x < 5 and y >= 5):
                    outside_area = True

            elif target_area == "SE":
                if not (x >= 5 and y >= 5):
                    outside_area = True


            tile = tiles[y][x]
            if tile == "LOCKED":
                continue

            base_score = None

            # 雑草
            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                if day >= StrategyConfig.GENERAL_PLANT_END_DAY:
                    continue

                base_score = 0

            #動物
            elif isinstance(tile, dict) and tile.get("animal"):
                if not tile.get("fed_today", False):
                    base_score = 100

                elif not tile.get("cared_today", False):
                    base_score = 75

                elif tile.get("yield_units", 0) > 0:
                    base_score = 60

                elif tile.get("fertilizer_available", 0) > 0:
                    base_score = 55

            # 植物
            elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop_name = tile.get("crop", "WHEAT")
                crop_age = day - tile.get("planted_day", day)
                harvest_age = get_harvest_age(crop_name)

                if crop_name == "STRAWBERRY":
                    if(
                        not tile.get("watered_today", True)
                        and tile.get("consecutive_unwatered", 0) >= 1
                    ):
                        base_score = 75

                    elif (crop_age >= harvest_age and tile.get("yield_units", 0) > 0):
                        base_score = 25

                    elif not tile.get("watered_today", True):
                        base_score = 50

                elif crop_age >= harvest_age:
                    if crop_name == "MELON":
                        base_score = 250

                    else:
                        base_score = 25

                elif not tile.get("watered_today", True):
                    if tile.get("consecutive_unwatered", 0) >= 1:
                        base_score = 75

                    else:
                        base_score = 50

            # 空き地
            elif tile is None and has_seeds:
                if day >= StrategyConfig.GENERAL_PLANT_END_DAY:
                    continue

                base_score = 50

            if base_score is None:
                continue

            movement_cost = abs(x - fx) + abs(y - fy)
            task_score = base_score - movement_cost

            if outside_area:
                task_score -= StrategyConfig.OUTSIDE_AREA_PENALTY

            if task_score > best_score:
                best_score = task_score
                best_target = (x, y)

    return best_target


def get_cow_coords(tiles):
    """牛がいる座標をまとめて返す。"""
    cow_coords = set()

    for y in range(len(tiles)):
        for x in range(len(tiles[0])):
            tile = tiles[y][x]

            if (
                isinstance(tile, dict)
                and tile.get("animal") == "COW"
            ):
                cow_coords.add((x, y))

    return cow_coords


def choose_plant_action(
    plant_allowed,
    melon_plant_allowed,
    strawberry_plant_allowed,
    remaining_melon_seeds,
    remaining_strawberry_seeds,
    remaining_carrot_seeds,
    remaining_wheat_seeds,
):
    """作物の優先順位に従って植付け行動を返す。"""
    action = None

    if (
        plant_allowed
        and melon_plant_allowed
        and remaining_melon_seeds > 0
    ):
        action = ["PLANT", "MELON"]
        remaining_melon_seeds -= 1

    elif (
        plant_allowed
        and strawberry_plant_allowed
        and remaining_strawberry_seeds > 0
    ):
        action = ["PLANT", "STRAWBERRY"]
        remaining_strawberry_seeds -= 1

    elif plant_allowed and remaining_carrot_seeds > 0:
        action = ["PLANT", "CARROT"]
        remaining_carrot_seeds -= 1

    elif plant_allowed and remaining_wheat_seeds > 0:
        action = ["PLANT", "WHEAT"]
        remaining_wheat_seeds -= 1

    return (
        action,
        remaining_melon_seeds,
        remaining_strawberry_seeds,
        remaining_carrot_seeds,
        remaining_wheat_seeds,
    )


def has_remaining_planting(
    plant_allowed,
    melon_plant_allowed,
    strawberry_plant_allowed,
    remaining_melon_seeds,
    remaining_strawberry_seeds,
    remaining_carrot_seeds,
    remaining_wheat_seeds,
):
    """現在植えられる種が残っているか返す。"""
    return (
        plant_allowed
        and (
            remaining_wheat_seeds > 0
            or remaining_carrot_seeds > 0
            or (
                melon_plant_allowed
                and remaining_melon_seeds > 0
            )
            or (
                strawberry_plant_allowed
                and remaining_strawberry_seeds > 0
            )
        )
    )


def move_to_work_target(
    tiles,
    actor_x,
    actor_y,
    has_seeds,
    day,
    claimed_targets,
    excluded_coords=None,
    target_area=None,
):
    """最適な作業対象へ移動し、対象がなければPASSする。"""
    if excluded_coords is None:
        excluded_coords = claimed_targets

    target = find_target_tile(
        tiles,
        actor_x,
        actor_y,
        has_seeds,
        day,
        excluded_coords,
        target_area,
    )

    if target is None:
        return ["PASS"]

    claimed_targets.add(target)
    move_dir = step_toward(
        actor_x,
        actor_y,
        target[0],
        target[1],
        tiles,
    )

    return [move_dir]

class HireController:
    """プレイヤーごとの作業員数をPASS率で調整する。"""

    def __init__(self):
        """プレイヤー別の雇用状態を初期化する。"""
        self.states = {}

    def update(self, player, day, step, unlocked_quads):
        """前日のPASS率から目標作業員数を更新する。"""
        if step == 0 or player not in self.states:
            self.states[player] = {
                "day": day,
                "pass_count": 0,
                "action_count": 0,
                "target_hands": StrategyConfig.INITIAL_TARGET_HANDS,
            }

            return self.states[player]

        state = self.states[player]

        if day != state["day"]:
            if state["action_count"] > 0:
                pass_rate = state["pass_count"] / state["action_count"]

                if pass_rate >= StrategyConfig.PASS_RATE_TO_DECREASE:
                    state["target_hands"] = max(
                        0,
                        state["target_hands"] - 1,
                    )

                elif pass_rate <= StrategyConfig.PASS_RATE_TO_INCREASE:
                    state["target_hands"] = min(
                        StrategyConfig.MAX_HANDS,
                        state["target_hands"] + 1,
                    )

            if len(unlocked_quads) >= StrategyConfig.TARGET_LAND_COUNT:
                state["target_hands"] = max(
                    StrategyConfig.MIN_LARGE_FARM_HANDS,
                    state["target_hands"],
                )

            state["day"] = day
            state["pass_count"] = 0
            state["action_count"] = 0

        return state

    def record(self, player, hands_actions):
        """現在ターンの作業員PASS数を記録する。"""
        state = self.states[player]

        for hand_action in hands_actions:
            state["action_count"] += 1

            if hand_action == ["PASS"]:
                state["pass_count"] += 1


# 雇用状態を対戦中に維持する。
hire_controller = HireController()

def build_market_actions(
    me,
    seeds,
    shed,
    inventories,
    day,
    step,
    melon_price,
    melon_stock,
    cow_count,
    target_hands,
    milk_price,
    unlocked_shops,
    max_market_orders,
):
    """現在の市場売買・雇用・土地購入ルールから注文一覧を作る。"""

    market = []

    money = me.get("money", 0)
    current_hands = me.get("hands", [])
    unlocked_quads = me.get(
        "unlocked_quadrants",
        ["NW"],
    )

    wheat_seeds = seeds.get("WHEAT", 0)
    melon_seeds = seeds.get("MELON", 0)

    strawberry_seeds = seeds.get("STRAWBERRY", 0)

    carrot_seeds = seeds.get("CARROT", 0)

    wheat_in_shed = shed.get("WHEAT", 0)
    melon_in_shed = shed.get("MELON", 0)

    milk_in_shed = shed.get("MILK", 0)

    strawberry_in_shed = shed.get("STRAWBERRY", 0)
    carrot_in_shed = shed.get("CARROT", 0)

    total_non_seed_items = sum(shed.values())

    for inventory in inventories:
        total_non_seed_items += sum(inventory.values())

    # 種を購入
    melon_planted_count = 0
    strawberry_planted_count = 0
    carrot_planted_count = 0

    for row in me["tiles"]:
        for tile in row:
            if(
                isinstance(tile, dict)
                and tile.get("kind") == "PLANT"
                and tile.get("crop") == "MELON"
            ):
                melon_planted_count += 1

            if(
                isinstance(tile, dict)
                and tile.get("kind") == "PLANT"
                and tile.get("crop") == "STRAWBERRY"
            ):
                strawberry_planted_count += 1

            if(
                isinstance(tile, dict)
                and tile.get("kind") == "PLANT"
                and tile.get("crop") == "CARROT"
            ):
                carrot_planted_count += 1

    strawberry_demand_shop_count = sum(
        1
        for shop in unlocked_shops
        if shop in StrategyConfig.STRAWBERRY_DEMAND_SHOPS
    )

    pet_cafe_count = sum(
        1
        for shop in unlocked_shops
        if shop == "PET_CAFE"
    )

    melon_total = melon_seeds + melon_planted_count
    melon_to_buy = max(StrategyConfig.MELON_TARGET_COUNT - melon_total, 0)

    if(
        day < StrategyConfig.MELON_PLANT_END_DAY
        and melon_to_buy > 0
        and money >= melon_to_buy * StrategyConfig.MELON_SEED_PRICE
    ):
        market.append(
            ["BUY_SEED", "MELON", melon_to_buy]
        )

    strawberry_total = (strawberry_seeds + strawberry_planted_count)
    strawberry_to_buy = max(
        StrategyConfig.STRAWBERRY_TARGET_COUNT - strawberry_total,
        0,
    )

    if (len(unlocked_quads) >= StrategyConfig.TARGET_LAND_COUNT
        and day < StrategyConfig.STRAWBERRY_PLANT_END_DAY
        and strawberry_demand_shop_count
        >= StrategyConfig.MIN_STRAWBERRY_DEMAND_SHOPS
        and strawberry_to_buy > 0
        and money >= strawberry_to_buy * StrategyConfig.STRAWBERRY_SEED_PRICE
    ):

        market.append(["BUY_SEED", "STRAWBERRY", strawberry_to_buy,])

    carrot_total = carrot_seeds + carrot_planted_count
    carrot_target_count = (pet_cafe_count * StrategyConfig.CARROT_TARGET_PER_PET_CAFE)
    carrot_to_buy = max(carrot_target_count - carrot_total, 0)

    if(
        day < StrategyConfig.GENERAL_PLANT_END_DAY
        and carrot_to_buy > 0
        and money >= carrot_to_buy * StrategyConfig.CARROT_SEED_PRICE
    ):
        market.append([
            "BUY_SEED",
            "CARROT",
            carrot_to_buy,
        ])

    if wheat_seeds == 0 and money >= StrategyConfig.WHEAT_SEED_PRICE:
        market.append([
            "BUY_SEED",
            "WHEAT",
            StrategyConfig.WHEAT_SEED_BUY_COUNT,
        ])


    # WHEAT売却
    if cow_count > 0:
        unfed_cow_count = 0

        for row in me["tiles"]:
            for tile in row:
                if(
                    isinstance(tile, dict)
                    and tile.get("animal") == "COW"
                    and not tile.get("fed_today", False)
                ):
                    unfed_cow_count += 1


        feed_worker_wheat = 0

        if len(inventories) > 0:
            feed_worker_wheat += inventories[0].get(
                "WHEAT",
                0,
            )

        if len(inventories) > 1:
            feed_worker_wheat += inventories[1].get(
                "WHEAT",
                0,
            )

        available_feed_wheat = (
            wheat_in_shed
            + feed_worker_wheat
        )

        target_feed_wheat = max(
            unfed_cow_count,
            StrategyConfig.MIN_FEED_WHEAT,
        )

        wheat_to_buy = max(
            target_feed_wheat
            - available_feed_wheat,
            0,
        )

        if wheat_to_buy > 0:
            market.append([
                "BUY_PRODUCT",
                "WHEAT",
                wheat_to_buy,
            ])

        excess_wheat = max(
            available_feed_wheat
            - target_feed_wheat,
            0,
        )

        wheat_to_sell = min(
            wheat_in_shed,
            excess_wheat,
        )

        if wheat_to_sell > 0:
            market.append([
                "SELL",
                "WHEAT",
                wheat_to_sell,
            ])

    elif wheat_in_shed > 0:
        market.append(["SELL", "WHEAT", wheat_in_shed])

    #CARROT売却
    if carrot_in_shed > 0:
        market.append([
            "SELL",
            "CARROT",
            carrot_in_shed,
        ])

    #MILK売却
    if milk_in_shed > 0:
        if should_sell_milk(milk_price, step, total_non_seed_items,):
            milk_to_sell = (
                milk_in_shed
                if step >= StrategyConfig.FINAL_SELL_STEP
                else min(milk_in_shed, StrategyConfig.MILK_SELL_BATCH)
            )

            market.append(["SELL", "MILK", milk_to_sell,])

    #STRAWBERRY売却
    if strawberry_in_shed > 0:
        market.append(["SELL", "STRAWBERRY", strawberry_in_shed])

    # MELON売却
    if melon_in_shed > 0:
        if should_sell_melon(
            melon_price,
            melon_stock,
            melon_in_shed,
            step,
        ):
            market.append(["SELL", "MELON", melon_in_shed])

    # HIRE後に追加される注文枠を事前に確保
    will_buy_land = (
        len(unlocked_quads) < StrategyConfig.TARGET_LAND_COUNT
        and money >= StrategyConfig.LAND_PRICE
    )

    cow_in_shed = shed.get("COW", 0)

    will_buy_cow = (
        len(unlocked_quads) >= 1
        and cow_count < StrategyConfig.TARGET_COW_COUNT
        and money >= StrategyConfig.COW_PRICE
    )

    reserved_market_orders = (
        int(will_buy_land) + int(will_buy_cow)
    )

    available_hire_slots = max(
        max_market_orders - len(market) - reserved_market_orders, 0,
    )

    hire_count = min(
        max(target_hands - len(current_hands), 0),
        available_hire_slots,
        StrategyConfig.MAX_HIRES_PER_TURN,
    )

    for _ in range(hire_count):
        market.append(["HIRE"])



    # 土地購入
    if will_buy_land:
        market.insert(0, ["BUY_LAND"],)

    #cowを飼う
    if will_buy_cow:
        market.append(["BUY_ANIMAL", "COW", 1])

    # 売却注文を最優先にする
    sell_orders = [
        order
        for order in market
        if order[0] == "SELL"

    ]

    other_orders = [
        order
        for order in market
        if order[0] != "SELL"
    ]

    market = (sell_orders + other_orders)


    return market


def agent(obs, config):
    """現在状態から農家・作業員・市場の行動を決定する。"""

    # 状態取得

    player = obs["player"]

    me = obs["farms"][player]
    private = obs["private"]

    tiles = me["tiles"]

    fx, fy = me["farmer"]
    farmer_tile = tiles[fy][fx]

    seeds = private.get("seeds", {})
    shed = private.get("shed", {})

    day = obs.get("day", 0)
    step = obs.get("step", 0)
    hour = obs.get("hour", 0)

    unlocked_shops = obs.get("town", {},).get("unlocked_shops", [],)
    current_hands = me.get("hands", [])
    milk_price = get_market_price(obs, "MILK",)

    melon_price = get_market_price(obs, "MELON",)
    melon_stock = obs["market"]["inventory"]["MELON"]

    wheat_seeds = seeds.get("WHEAT", 0)
    melon_seeds = seeds.get("MELON", 0)
    strawberry_seeds = seeds.get("STRAWBERRY", 0)
    carrot_seeds = seeds.get("CARROT", 0)

    remaining_wheat_seeds = wheat_seeds
    remaining_melon_seeds = melon_seeds
    remaining_strawberry_seeds = strawberry_seeds
    remaining_carrot_seeds = carrot_seeds

    melon_plant_allowed = day < StrategyConfig.MELON_PLANT_END_DAY
    strawberry_plant_allowed = day < StrategyConfig.STRAWBERRY_PLANT_END_DAY

    wheat_in_shed = shed.get("WHEAT", 0)

    inventories = private.get("inventories", [])

    farmer_inventory = (
        inventories[0]
        if len(inventories) > 0
        else {}
    )
    farmer_cow = farmer_inventory.get("COW", 0)

    farmer_wheat = farmer_inventory.get("WHEAT", 0)
    farmer_fertilizer = farmer_inventory.get("FERTILIZER", 0)

    farmer_milk = farmer_inventory.get("MILK", 0)

    pasture_count = 0

    for row in tiles:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == "PASTURE":
                pasture_count += 1

    cow_count = 0

    for row in tiles:
        for tile in row:
            if (
                isinstance(tile, dict) and tile.get("animal") == "COW"
            ):
                cow_count += 1
    cow_count += shed.get("COW", 0)
    cow_count += farmer_cow

    hire_state = hire_controller.update(
        player,
        day,
        step,
        me.get("unlocked_quadrants", ["NW"],),
    )

    target_hands = hire_state["target_hands"]

    urgent_unwatered_count = 0

    for row in tiles:
        for tile in row:
            if (
                isinstance(tile, dict)
                and tile.get("kind") == "PLANT"
                and not tile.get("watered_today", True)
                and tile.get("consecutive_unwatered", 0) >= 1
            ):
                urgent_unwatered_count += 1

    plant_allowed = (urgent_unwatered_count == 0)


    # 市場

    market = build_market_actions(
        me,
        seeds,
        shed,
        inventories,
        day,
        step,
        melon_price,
        melon_stock,
        cow_count,
        target_hands,
        milk_price,
        unlocked_shops,
        config.get("maxMarketOrdersPerTurn", 10,),
    )

    # メイン農家

    farmer_action = None
    cow_in_shed = shed.get("COW", 0)

    if step >= StrategyConfig.FINAL_SELL_STEP and farmer_milk > 0:
        if (fx, fy) == StrategyConfig.SHED_COORD:
            farmer_action = ["PLACE", "MILK", farmer_milk,]

        else:
            move_dir = step_toward(
                fx,
                fy,
                StrategyConfig.SHED_COORD[0],
                StrategyConfig.SHED_COORD[1],
                tiles,
            )
            farmer_action = [move_dir]



    elif farmer_cow > 0:
        if (
            isinstance(farmer_tile, dict)
            and farmer_tile.get("kind") == "PASTURE"
            and not farmer_tile.get("animal")
        ):
            farmer_action = ["PLACE","COW"]

        else:
            pasture_target = find_empty_pasture(tiles, fx, fy)

            if pasture_target is not None:
                move_dir = step_toward(
                    fx,
                    fy,
                    pasture_target[0],
                    pasture_target[1],
                    tiles,
                )

                farmer_action = [move_dir]

    elif(
        cow_in_shed > 0
        and find_empty_pasture(tiles, fx, fy) is not None
        and hour == 0
    ):
        farmer_action = ["PICKUP", "COW", 1,]

    elif(
        cow_count > 0
        and farmer_wheat == 0
        and wheat_in_shed > 0
        and hour == 0
    ):
        farmer_action = ["PICKUP", "WHEAT", 1,]


    elif farmer_tile is None:
        if (
            cow_in_shed > 0
            and pasture_count < StrategyConfig.TARGET_PASTURE_COUNT
        ):
            farmer_action = ["BUILD_PASTURE",]

        else:
            (
                farmer_action,
                remaining_melon_seeds,
                remaining_strawberry_seeds,
                remaining_carrot_seeds,
                remaining_wheat_seeds,
            ) = choose_plant_action(
                plant_allowed,
                melon_plant_allowed,
                strawberry_plant_allowed,
                remaining_melon_seeds,
                remaining_strawberry_seeds,
                remaining_carrot_seeds,
                remaining_wheat_seeds,
            )

    else:
        farmer_action = get_tile_action(farmer_tile, day,)

    # メイン農家の移動

    claimed_targets = set()

    if(farmer_action is None and farmer_fertilizer > 0):
        if(
            isinstance(farmer_tile, dict)
            and farmer_tile.get("kind") == "PLANT"
            and farmer_tile.get("fertilized_until_day", -1,) < day
        ):
            farmer_action = ["FERTILIZE"]

        else:
            fertilizer_target = find_fertilize_target(tiles, fx, fy, day)

            if fertilizer_target is not None:
                move_dir = step_toward(
                    fx, fy,
                    fertilizer_target[0],
                    fertilizer_target[1],
                    tiles,
                )

                farmer_action = [move_dir]

    if farmer_action is None:
        remaining_has_seeds = has_remaining_planting(
            plant_allowed,
            melon_plant_allowed,
            strawberry_plant_allowed,
            remaining_melon_seeds,
            remaining_strawberry_seeds,
            remaining_carrot_seeds,
            remaining_wheat_seeds,
        )

        farmer_action = move_to_work_target(
            tiles,
            fx,
            fy,
            remaining_has_seeds,
            day,
            claimed_targets,
        )
    else:
        claimed_targets.add(
            (fx, fy)
        )


    # 作業員

    hands_actions = []

    for hand_index, hand in enumerate(current_hands):
        hx, hy = hand

        hand_inventory = (
            inventories[hand_index + 1]
            if len(inventories) > hand_index + 1
            else {}
        )

        hand_wheat = hand_inventory.get("WHEAT", 0)

        hand_tile = tiles[hy][hx]

        if hand_index == 0:

            if(
                cow_count > 0
                and hand_wheat == 0
                and wheat_in_shed > 0
            ):
                if (hx, hy) == StrategyConfig.SHED_COORD:
                    hand_action = ["PICKUP", "WHEAT", 1,]

                else:
                    move_dir = step_toward(
                        hx,
                        hy,
                        StrategyConfig.SHED_COORD[0],
                        StrategyConfig.SHED_COORD[1],
                        tiles,
                    )
                    hand_action = [move_dir]

            elif(
                isinstance(hand_tile, dict)
                and hand_tile.get("animal") == "COW"
            ):
                hand_action = get_tile_action(
                    hand_tile,
                    day,
                    False,
                )

                if hand_action == ["HARVEST"] and (hx, hy) in claimed_targets:
                    hand_action = None

                if hand_action is None:
                    cow_target = find_cow_target(
                        tiles,
                        hx,
                        hy,
                        day,
                        False,
                    )

                    if cow_target is not None:
                        move_dir = step_toward(
                            hx,
                            hy,
                            cow_target[0],
                            cow_target[1],
                            tiles,
                        )
                        hand_action = [move_dir]

                    else:
                        hand_action = None

            else:
                cow_target = find_cow_target(
                    tiles,
                    hx,
                    hy,
                    day,
                    False,
                )

                if cow_target is not None:
                    move_dir = step_toward(
                        hx,
                        hy,
                        cow_target[0],
                        cow_target[1],
                        tiles,
                    )
                    hand_action = [move_dir]

                else:
                    hand_action = None

            if hand_action is None:
                if hand_tile is None:
                    (
                        hand_action,
                        remaining_melon_seeds,
                        remaining_strawberry_seeds,
                        remaining_carrot_seeds,
                        remaining_wheat_seeds,
                    ) = choose_plant_action(
                        plant_allowed,
                        melon_plant_allowed,
                        strawberry_plant_allowed,
                        remaining_melon_seeds,
                        remaining_strawberry_seeds,
                        remaining_carrot_seeds,
                        remaining_wheat_seeds,
                    )

                elif not(
                    isinstance(hand_tile, dict)
                    and hand_tile.get("animal")
                ):
                    hand_action = get_tile_action(hand_tile, day,)

                    if(hand_action == ["HARVEST"] and (hx, hy) in claimed_targets):
                        hand_action = None


            if hand_action is None:
                cow_coords = get_cow_coords(tiles)

                remaining_has_seeds = has_remaining_planting(
                    plant_allowed,
                    melon_plant_allowed,
                    strawberry_plant_allowed,
                    remaining_melon_seeds,
                    remaining_strawberry_seeds,
                    remaining_carrot_seeds,
                    remaining_wheat_seeds,
                )

                hand_area = get_hand_area(hand_index,)

                hand_action = move_to_work_target(
                    tiles,
                    hx,
                    hy,
                    remaining_has_seeds,
                    day,
                    claimed_targets,
                    excluded_coords=claimed_targets | cow_coords,
                    target_area=hand_area,
                )


        else:

            if hand_tile is None:
                (
                    hand_action,
                    remaining_melon_seeds,
                    remaining_strawberry_seeds,
                    remaining_carrot_seeds,
                    remaining_wheat_seeds,
                ) = choose_plant_action(
                    plant_allowed,
                    melon_plant_allowed,
                    strawberry_plant_allowed,
                    remaining_melon_seeds,
                    remaining_strawberry_seeds,
                    remaining_carrot_seeds,
                    remaining_wheat_seeds,
                )

            elif(
                isinstance(hand_tile, dict)
                and hand_tile.get("animal")
            ):
                hand_action = None

            else:
                hand_action = get_tile_action(
                    hand_tile,
                    day,
                )

            if (hand_action == ["HARVEST"] and (hx, hy) in claimed_targets):
                hand_action = None

            if hand_action is None:
                cow_coords = get_cow_coords(tiles)

                remaining_has_seeds = has_remaining_planting(
                    plant_allowed,
                    melon_plant_allowed,
                    strawberry_plant_allowed,
                    remaining_melon_seeds,
                    remaining_strawberry_seeds,
                    remaining_carrot_seeds,
                    remaining_wheat_seeds,
                )

                hand_area = get_hand_area(hand_index,)

                hand_action = move_to_work_target(
                    tiles,
                    hx,
                    hy,
                    remaining_has_seeds,
                    day,
                    claimed_targets,
                    excluded_coords=claimed_targets | cow_coords,
                    target_area=hand_area,
                )


        if hand_action is not None:
            claimed_targets.add(
                (hx, hy)
            )

        hands_actions.append(
            hand_action
        )

    hire_controller.record(
        player,
        hands_actions,
    )


    # 出力

    return {
        "farmer": farmer_action,
        "hands": hands_actions,
        "market": market,
    }