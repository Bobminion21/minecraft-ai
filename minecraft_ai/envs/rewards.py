"""Tech-tree reward shaping for Minecraft.

Adds intermediate rewards for crafting milestones to help the agent
learn the long sequence: log -> planks -> crafting_table -> sticks ->
wooden_pickaxe -> cobblestone -> stone_pickaxe -> iron_ore -> furnace ->
iron_ingot -> iron_pickaxe -> diamond
"""

TECH_TREE_REWARDS = {
    # Item name -> (reward, required_count)
    "log": (1.0, 1),
    "planks": (2.0, 1),
    "crafting_table": (4.0, 1),
    "stick": (2.0, 1),
    "wooden_pickaxe": (8.0, 1),
    "cobblestone": (4.0, 1),
    "stone_pickaxe": (16.0, 1),
    "iron_ore": (32.0, 1),
    "furnace": (32.0, 1),
    "iron_ingot": (64.0, 1),
    "iron_pickaxe": (128.0, 1),
    "diamond": (1024.0, 1),
}


class RewardShaper:
    """Tracks inventory and gives bonus rewards for new item acquisitions."""

    def __init__(self):
        self.prev_inventory = {}

    def shape(self, inventory: dict) -> float:
        """Compute shaped reward based on inventory changes.

        Args:
            inventory: dict of {item_name: count}

        Returns:
            bonus reward for this step
        """
        bonus = 0.0
        for item, (reward, threshold) in TECH_TREE_REWARDS.items():
            prev_count = self.prev_inventory.get(item, 0)
            curr_count = inventory.get(item, 0)
            if curr_count >= threshold and prev_count < threshold:
                bonus += reward

        self.prev_inventory = dict(inventory)
        return bonus

    def reset(self):
        self.prev_inventory = {}
