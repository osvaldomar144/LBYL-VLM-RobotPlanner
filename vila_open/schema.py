# vila_open/schema.py

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import json

@dataclass
class Action:
    """
    Azione 'Pilota' per il controllo diretto tramite robot_vlm_lib.
    """
    primitive: str          # "base", "arm", "torso", "gripper"
    params: List[float]     # es: [0.5, 0, 0] oppure [1.0]
    
    # Campo opzionale per farci dire dalla VLM cosa sta cercando di fare (Reasoning)
    reasoning: Optional[str] = None 

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primitive": self.primitive,
            "params": self.params,
            "reasoning": self.reasoning
        }

@dataclass
class Plan:
    """
    In modalità Pilota, il 'Plan' è solitamente composto da UNA sola azione immediata,
    perché la VLM ricalcola tutto al frame successivo.
    """
    plan: List[Action]

    def to_dict(self) -> Dict[str, Any]:
        return {"plan": [a.to_dict() for a in self.plan]}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Plan":
        actions: List[Action] = []
        for a in d.get("plan", []):
            primitive = a.get("primitive")
            params = a.get("params")
            reasoning = a.get("reasoning")
            
            # Gestione robusta dei parametri (es. se la VLM manda stringhe per il gripper)
            if primitive == "gripper" and isinstance(params, list):
                if params[0] == "close": params = [1.0]
                elif params[0] == "open": params = [-1.0]

            if primitive and params is not None:
                actions.append(Action(primitive=primitive, params=params, reasoning=reasoning))
        
        return Plan(plan=actions)

    @staticmethod
    def from_json(s: str) -> "Plan":
        return Plan.from_dict(json.loads(s))