import numpy as np
import robosuite.utils.transform_utils as T
from .common import SimUtils

class OracleController:
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        self.action_dim = self.env.action_spec[0].shape[0]
        
        # --- MAPPATURA PANDA OMRON (12 DIM) ---
        self.idx_arm = slice(0, 6)
        self.idx_torso = 6
        self.idx_base = slice(7, 10) # X, Y, Rot
        self.idx_grip = slice(10, 12)
        
        print(f"[Controller] READY. Mode: Navigation -> Untuck -> Pick.")
        
        # Posa "Grip Down" standard
        self.quat_down = np.array([-0.03, 0.99, 0.00, -0.02]) 

    def execute_primitive(self, primitive, obj_name):
        primitive = primitive.lower()
        if primitive == "maps": primitive = "navigate"
        
        # 1. Trova l'oggetto
        target_pos = SimUtils.get_object_pose(self.sim, obj_name)
        
        # 2. Calcola distanza iniziale
        base_pos, _ = SimUtils.get_base_pose(self.sim)
        dist_2d = np.linalg.norm(target_pos[:2] - base_pos[:2])
        
        print(f"\n[Controller] CMD: {primitive.upper()} | Target Dist: {dist_2d:.2f}m")

        # --- FASE 1: NAVIGAZIONE ---
        # Naviga se il target è lontano più di 75cm o se richiesto esplicitamente
        if primitive == "navigate" or dist_2d > 0.75:
            print(" -> [Phase 1] Navigazione Base...")
            # Ci fermiamo a 65cm per avere spazio di manovra
            yield from self._navigate_to_target(target_pos, stop_dist=0.65)
            
            # Pausa di stabilizzazione
            for _ in range(20): yield self._get_idle_action()
            
            if primitive == "navigate": return

        # --- FASE 2: PREPARAZIONE BRACCIO (Untuck) ---
        # Se dobbiamo manipolare, portiamo il braccio davanti al robot
        if primitive in ["pick", "place"]:
            print(" -> [Phase 2] Untuck Arm (Posizione Neutra)...")
            yield from self._untuck_arm()

        # --- FASE 3: MANIPOLAZIONE ---
        if primitive == "pick":
            print(" -> [Phase 3] Esecuzione Pick...")
            yield from self._pick_sequence(target_pos)
            
        elif primitive == "place":
            yield from self._place_sequence(target_pos)

    # ------------------------------------------------------------------
    # UTILS
    # ------------------------------------------------------------------
    def _get_idle_action(self):
        """Azione neutra: tutto fermo."""
        action = np.zeros(self.action_dim)
        action[self.idx_grip] = -1.0 # Gripper aperto
        return action

    # ------------------------------------------------------------------
    # NAVIGAZIONE
    # ------------------------------------------------------------------
    def _navigate_to_target(self, target_pos, stop_dist):
        kp_lin = 1.2
        kp_rot = 2.0
        
        print(f"   [Nav] Start. Distanza attuale: {np.linalg.norm(target_pos[:2] - SimUtils.get_base_pose(self.sim)[0][:2]):.2f}m")
        
        for i in range(500): 
            curr_pos, curr_quat = SimUtils.get_base_pose(self.sim)
            
            diff = target_pos[:2] - curr_pos[:2]
            dist = np.linalg.norm(diff)
            
            if dist <= stop_dist:
                print("   [Nav] Arrivati a destinazione.")
                break 
            
            # Calcolo angolo target
            target_angle = np.arctan2(diff[1], diff[0])
            
            # FIX: Uso get_safe_euler per evitare il crash
            curr_euler = SimUtils.get_safe_euler(curr_quat)
            curr_yaw = curr_euler[2] 
            
            # Errore angolare normalizzato
            angle_err = target_angle - curr_yaw
            angle_err = (angle_err + np.pi) % (2 * np.pi) - np.pi
            
            action = np.zeros(self.action_dim)
            
            # Logica Turn-Move: Se l'angolo è grande, ruota. Se è piccolo, avanza.
            if abs(angle_err) > 0.15: 
                # Rotazione in place
                action[9] = np.clip(angle_err * kp_rot, -1.0, 1.0)
            else:
                # Avanzamento + micro correzione rotazione
                action[7] = np.clip(dist * kp_lin, -1.0, 1.0)
                action[9] = np.clip(angle_err * kp_rot, -0.5, 0.5)

            action[self.idx_grip] = -1.0
            
            yield action
            
        for _ in range(10): yield self._get_idle_action()

    # ------------------------------------------------------------------
    # MANIPOLAZIONE (OSC)
    # ------------------------------------------------------------------
    def _untuck_arm(self):
        """Porta il braccio in una posa di 'Carry' davanti al robot."""
        base_pos, base_quat = SimUtils.get_base_pose(self.sim)
        base_mat = T.quat2mat(base_quat)
        
        # Target relativo alla base: 60cm avanti, 1.1m alto
        offset = np.array([0.60, 0.0, 0.0]) 
        target_world = base_pos + (base_mat @ offset)
        target_world[2] = 1.15
        
        yield from self._move_arm_osc(target_world, self.quat_down, grasp=-1, steps=80, precision=0.15, debug_name="Untuck")

    def _move_arm_osc(self, target_pos, target_quat, grasp, steps=50, precision=0.02, debug_name="Move"):
        kp_pos = 5.0
        kp_ori = 2.0
        
        for i in range(steps):
            try:
                site_id, curr_pos, curr_quat = SimUtils.get_eef_data(self.sim)
            except ValueError: break 

            err_pos = target_pos - curr_pos
            if np.dot(target_quat, curr_quat) < 0.0: target_quat = -target_quat
            err_ori = T.get_orientation_error(target_quat, curr_quat)
            
            if i > 10 and np.linalg.norm(err_pos) < precision:
                break

            # TRASFORMAZIONE FONDAMENTALE: World Frame -> Robot Base Frame
            # Il controller OSC vuole i delta relativi alla base del robot
            base_pos, base_quat = SimUtils.get_base_pose(self.sim)
            base_mat = T.quat2mat(base_quat)
            
            # v_robot = R_base_inv * v_world
            action_pos = base_mat.T @ err_pos
            action_ori = base_mat.T @ err_ori
            
            action = np.zeros(self.action_dim)
            action[self.idx_arm] = np.concatenate([
                np.clip(action_pos * kp_pos, -1, 1),
                np.clip(action_ori * kp_ori, -0.5, 0.5)
            ])
            action[self.idx_torso] = 0.0
            action[self.idx_base] = 0.0 
            action[self.idx_grip] = grasp 
            
            yield action

    def _pick_sequence(self, obj_pos):
        # Hover sopra
        hover = obj_pos.copy(); hover[2] += 0.25
        yield from self._move_arm_osc(hover, self.quat_down, -1, 60, 0.03, "Hover")
        
        # Scendi
        approach = obj_pos.copy(); approach[2] += 0.005 # Molto vicino
        yield from self._move_arm_osc(approach, self.quat_down, -1, 60, 0.01, "Approach")
        
        print("   [Pick] Grasping...")
        # Chiudi pinza (attesa)
        for _ in range(30):
            yield from self._move_arm_osc(approach, self.quat_down, 1, 1, 0.05, "Grasp")
            
        # Solleva
        yield from self._move_arm_osc(hover, self.quat_down, 1, 60, 0.05, "Lift")

    def _place_sequence(self, target_pos):
        hover = target_pos.copy(); hover[2] += 0.25
        place = target_pos.copy(); place[2] += 0.05
        yield from self._move_arm_osc(hover, self.quat_down, 1, 60, 0.05, "PlHover")
        yield from self._move_arm_osc(place, self.quat_down, 1, 60, 0.02, "PlLower")
        for _ in range(30):
            yield from self._move_arm_osc(place, self.quat_down, -1, 1, 0.05, "PlRelease")
        yield from self._move_arm_osc(hover, self.quat_down, -1, 40, 0.05, "PlRetract")