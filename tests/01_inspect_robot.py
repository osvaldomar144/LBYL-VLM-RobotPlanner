import robocasa
import numpy as np

def get_controller_dim(controller):
    """
    Funzione helper per estrarre la dimensione in modo sicuro
    provando diversi attributi noti di Robosuite.
    """
    # Tentativo 1: action_dim (Standard)
    if hasattr(controller, "action_dim"):
        return controller.action_dim
    
    # Tentativo 2: control_dim (OSC Controllers)
    if hasattr(controller, "control_dim"):
        return controller.control_dim
        
    # Tentativo 3: dai limiti di input
    if hasattr(controller, "command_input_limits"):
        # limits è solitamente una tupla (low, high), prendiamo la lunghezza di low
        return len(controller.command_input_limits[0])
        
    return "N/A"

def inspect_robot():
    print("Inizializzazione ambiente per ispezione (senza camere)...")
    
    # Disabilitiamo tutto il rendering
    env = robocasa.make(
        env_name="PnPCounterToSink",
        robots="PandaMobile",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False
    )
    env.reset()
    
    robot = env.robots[0]
    total_action_dim = env.action_spec[0].shape[0]
    
    print(f"\n\n{'='*50}")
    print(f"REPORT ROBOT: {robot.name}")
    print(f"Classe Python: {type(robot).__name__}")
    print(f"Action Dimension Totale: {total_action_dim}")
    print(f"{'='*50}")
    
    # Logica per Robot Compositi (WheeledRobot)
    if hasattr(robot, "part_controllers"):
        print("\n--- MAPPATURA PARTI (COMPOSITE) ---")
        current_idx = 0
        
        # Iteriamo sulle parti
        for part_name, controller in robot.part_controllers.items():
            # Recupera dimensione in modo sicuro
            dim = get_controller_dim(controller)
            
            # Calcolo indici start/end
            if isinstance(dim, int):
                idx_start = current_idx
                idx_end = current_idx + dim
                current_idx += dim
                range_str = f"[{idx_start} : {idx_end}]"
            else:
                range_str = "???"

            print(f"> PART: '{part_name.upper()}'")
            print(f"  - Classe: {type(controller).__name__}")
            print(f"  - Nome:   {getattr(controller, 'name', 'N/A')}")
            print(f"  - Indici: {range_str}")
            print(f"  - Dims:   {dim}")
            print("-" * 30)
            
    else:
        print(f"Controller Config: {getattr(robot, 'controller_config', 'Non Trovato')}")

    # Vediamo i limiti numerici per conferma
    low, high = env.action_spec
    print(f"\n--- LIMITI AZIONI (Primi {total_action_dim}) ---")
    print(f"Low:  {low[:total_action_dim]}")
    print(f"High: {high[:total_action_dim]}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    inspect_robot()