import sys
import gymnasium as gym

# 1. IMPORTA IL TUO AMBIENTE CUSTOM
# Questo è il passaggio fondamentale: importando il file,
# Python esegue il decoratore @register_env e registra l'ambiente.
import my_custom_franka_env 

# 2. IMPORTA IL TOOL DI MANISKILL
from mani_skill.examples import demo_manual_control

if __name__ == "__main__":
    # Esegue la funzione main dello script di teleoperazione
    # Passeremo gli argomenti da riga di comando direttamente a lui
    demo_manual_control.main()