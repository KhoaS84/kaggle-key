"""
Starter Agent for Kaggriculture Competition.
"""

def agent(obs):
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    fx, fy = me["farmer"]
    tile = me["tiles"][fy][fx]
    
    market = []
    
    # 1. Market Buy: Buy Wheat Seed if none in inventory and have money
    wheat_seeds = private["seeds"].get("WHEAT", 0)
    if wheat_seeds == 0 and me["money"] >= 10:
        market.append(["BUY_SEED", "WHEAT", 1])
        
    # 2. Market Sell: Sell harvested wheat stored in shed
    wheat_in_shed = private["shed"].get("WHEAT", 0)
    if wheat_in_shed > 0:
        market.append(["SELL", "WHEAT", wheat_in_shed])
        
    # 3. Farmer Action: Plant, Harvest, Water, or Pass
    if tile is None and wheat_seeds > 0:
        return {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": market}
        
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop_age = obs["day"] - tile["planted_day"]
        # Wheat first_yield_day = 2, peak = 4
        if crop_age >= 2:
            return {"farmer": ["HARVEST"], "hands": [], "market": market}
        if not tile.get("watered_today", False):
            return {"farmer": ["WATER"], "hands": [], "market": market}
            
    return {"farmer": ["PASS"], "hands": [], "market": market}
