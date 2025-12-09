# vila_open/schema.py

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import json


@dataclass
class Action:
    """
    Azione simbolica di alto livello (stile ViLa).
    """
    step_id: int
    primitive: str               # es: "pick", "place", "open", ...
    object: Optional[str] = None
    from_location: Optional[str] = None
    to_location: Optional[str] = None

    # conterrà campi extra come "destination", "button", "knob_id", ecc.
    extra_args: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        # campi base
        base = {
            "step_id": self.step_id,
            "primitive": self.primitive,
            "object": self.object,
            "from_location": self.from_location,
            "to_location": self.to_location,
        }
        # aggiungi gli argomenti extra (es. destination, button...)
        if self.extra_args:
            base.update(self.extra_args)

        # rimuovi i None
        return {k: v for k, v in base.items() if v is not None}


@dataclass
class Plan:
    """
    Piano = lista ordinata di azioni.
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
            # campi noti
            step_id = a.get("step_id")
            primitive = a.get("primitive")
            obj = a.get("object")
            from_loc = a.get("from_location")
            to_loc = a.get("to_location")

            # tutto ciò che non è tra i campi noti va in extra_args
            known_keys = {"step_id", "primitive", "object", "from_location", "to_location"}
            extra = {k: v for k, v in a.items() if k not in known_keys}

            actions.append(
                Action(
                    step_id=step_id,
                    primitive=primitive,
                    object=obj,
                    from_location=from_loc,
                    to_location=to_loc,
                    extra_args=extra if extra else None,
                )
            )
        return Plan(plan=actions)

    @staticmethod
    def from_json(s: str) -> "Plan":
        return Plan.from_dict(json.loads(s))