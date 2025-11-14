# src/config.py
import os
import yaml
import logging

logging.basicConfig(level=logging.INFO)

def load_config(config_path='config.yaml'):
    """
    Carrega o config.yaml da raiz do projeto.
    Retorna um dict com as configs.
    """
    try:
        # Encontra a raiz do projeto (funciona de src/)
        project_root = os.path.dirname(os.path.dirname(__file__))
        full_path = os.path.join(project_root, config_path)
        
        with open(full_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Config carregado de {full_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Arquivo {config_path} não encontrado em {project_root}")
        raise  # Ou retorne defaults: return {}
    except yaml.YAMLError as e:
        logging.error(f"Erro ao parsear YAML: {e}")
        raise

# Opcional: Carregue globalmente se quiser acessar como CONFIG
CONFIG = load_config()  # Mas evite globals se possível; use a função