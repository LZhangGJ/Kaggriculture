"""CPU-only trusted referee smoke. Does not validate Docker isolation."""
import json
from tools.arena.worker import official_game, AgentFailure


if __name__ == "__main__":
    contract = {"version":"1.32.7","episode_steps":720,"game_timeout_seconds":120}
    def idle(obs, config):
        return {"farmer":["WAIT"]}
    result = official_game([idle,idle], {"seed":123}, contract)
    assert result["terminal"] and result["steps"] == 720 and result["cash"] == [3000.,3000.]
    print(json.dumps(result))
