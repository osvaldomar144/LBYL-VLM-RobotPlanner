import numpy as np
import robosuite.utils.transform_utils as T

class SimUtils:
    """
    Gestisce l'interazione con MuJoCo e fornisce utility matematiche sicure.
    Include uno SCANNER avanzato per trovare oggetti con nomi procedurali.
    """

    @staticmethod
    def get_safe_euler(quat):
        mat = T.quat2mat(quat)
        return T.mat2euler(mat)

    @staticmethod
    def get_eef_data(sim):
        """Trova la posa dell'End Effector (Pinza)."""
        # (Codice invariato: cerca i vari nomi possibili del gripper)
        candidates = ["gripper0_grip_site", "grip_site", "right_gripper", "robot0_eef", "ee_site"]
        site_id = None
        for name in candidates:
            try:
                site_id = sim.model.site_name2id(name)
                break
            except ValueError: continue
            
        if site_id is None:
             for i in range(sim.model.nsite):
                name = sim.model.site_id2name(i)
                if not name: continue
                n = name.lower()
                if ("grip" in n or "eef" in n) and not any(x in n for x in ["ft_", "imu", "cyl", "sensor"]):
                    site_id = i
                    break
        
        if site_id is None:
            raise ValueError("[SimUtils] CRITICO: EEF Site non trovato!")

        pos = sim.data.site_xpos[site_id].copy()
        mat = sim.data.site_xmat[site_id].reshape(3, 3)
        quat = T.mat2quat(mat) 
        return site_id, pos, quat

    @staticmethod
    def get_base_pose(sim):
        """Ritorna posa della base."""
        body_id = None
        try: body_id = sim.model.body_name2id("robot0_base")
        except: 
            try: body_id = sim.model.body_name2id("base_link")
            except: pass
            
        if body_id is not None:
            pos = sim.data.body_xpos[body_id].copy()
            quat = sim.data.body_xquat[body_id].copy()
            return pos, quat
        return np.zeros(3), np.array([0,0,0,1])

    @staticmethod
    def scan_scene_objects(sim):
        """
        Scansiona TUTTI i body della scena e filtra quelli che sembrano oggetti.
        Ritorna una lista di tuple (nome, posizione).
        """
        objects = []
        for i in range(sim.model.nbody):
            name = sim.model.body_id2name(i)
            if not name: continue
            
            # Filtri per escludere il robot e l'ambiente statico
            ignored_terms = ["robot", "base", "link", "wall", "floor", "visual", "collision", "root", "site"]
            name_low = name.lower()
            
            if any(term in name_low for term in ignored_terms):
                continue
                
            # RoboCasa usa spesso prefissi come 'obj_' o nomi di categorie
            # Accettiamo tutto ciò che ha 'obj', 'veg', 'grocer', 'food', 'container'
            # O anche nomi generici se non contengono i termini ignorati
            if "obj" in name_low or "veg" in name_low or "grocer" in name_low or "food" in name_low:
                pos = sim.data.body_xpos[i].copy()
                objects.append((name, pos))
                
        return objects

    @staticmethod
    def get_object_pose(sim, obj_name):
        """
        Trova la posa target usando lo SCANNER.
        """
        print(f"\n--- [SimUtils] ANALISI SCENA PER: '{obj_name}' ---")
        
        # 1. Ottieni lista di TUTTI gli oggetti candidati
        candidates = SimUtils.scan_scene_objects(sim)
        
        if not candidates:
            print(" > ATTENZIONE: Nessun oggetto interagibile rilevato nella scena!")
            return np.array([0.5, -0.5, 0.9]) # Fallback generico
            
        print(f" > Oggetti rilevati nel simulatore ({len(candidates)}):")
        for name, pos in candidates:
            print(f"   - '{name}' @ {pos}")
            
        # 2. Logica di Matching
        obj_name_clean = obj_name.lower().replace(" ", "_")
        search_terms = [obj_name_clean]
        
        # Sinonimi comuni per RoboCasa
        if "pepper" in obj_name_clean: search_terms.append("vegetable")
        if "milk" in obj_name_clean: search_terms.append("grocer")
        if "bread" in obj_name_clean: search_terms.append("grocer")
        
        best_match_name = None
        best_match_pos = None
        
        # Cerchiamo prima il match esatto, poi per categoria
        for term in search_terms:
            for name, pos in candidates:
                if term in name.lower():
                    # Preferiamo i body che finiscono con "_main" (sono quelli fisici solitamente)
                    is_main = "_main" in name.lower()
                    if best_match_name is None or (is_main and "main" not in best_match_name):
                        best_match_name = name
                        best_match_pos = pos
            
            if best_match_pos is not None:
                break # Trovato un match per questo termine
        
        # 3. Risultato
        if best_match_pos is not None:
            print(f" > MATCH TROVATO: '{best_match_name}' (Termine: '{term}')")
            print(f" > COORDINATE: {best_match_pos}")
            return best_match_pos

        # 4. Fallback intelligente: Prendi il primo oggetto che contiene "obj"
        print(f" > NESSUN MATCH DIRETTO. Cerco un oggetto generico...")
        for name, pos in candidates:
            if "obj" in name.lower() and "main" in name.lower():
                print(f" > Fallback Auto: Uso '{name}'")
                return pos
                
        # Fallback totale
        print(" > FAIL TOTALE. Uso coordinate fallback 'counter'.")
        return np.array([0.0, -0.60, 0.90])

    @staticmethod
    def world_to_robot_velocity(vel_world, base_quat):
        mat = T.quat2mat(base_quat)
        return mat.T @ vel_world