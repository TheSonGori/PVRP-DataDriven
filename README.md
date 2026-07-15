# PVRP-RL: Resolución del Problema de Ruteo de Vehículos Periódico mediante Aprendizaje por Refuerzo

Repositorio asociado a la memoria de título **"Resolución del Problema de Ruteo de Vehículos Periódico (PVRP) mediante un Enfoque Data-Driven con Aprendizaje por Refuerzo"**.

**Autora:** Javiera Elena Gutiérrez Abarca

---

## Descripción

El **Periodic Vehicle Routing Problem (PVRP)** es una extensión del clásico Vehicle Routing Problem (VRP) en la que cada cliente debe ser visitado con una frecuencia determinada a lo largo de un horizonte de planificación de varios días. El problema integra dos decisiones acopladas: la asignación temporal de visitas (¿qué días visitar a cada cliente?) y la construcción espacial de rutas (¿en qué orden recorrerlos?). Su naturaleza combinatoria lo clasifica como **NP-hard**, lo que impide resolverlo de manera exacta para instancias de tamaño realista.

Este proyecto propone un enfoque **data-driven** basado en **Aprendizaje por Refuerzo (RL)** para abordar el PVRP. Un agente entrenado con **MaskablePPO** (sb3-contrib) aprende, por interacción con un entorno simulado compatible con Gymnasium, a construir soluciones de manera secuencial y factible por construcción (Action Masking), sin depender de reglas diseñadas manualmente. El desempeño del modelo se compara contra dos métodos de referencia ampliamente usados en la literatura: una heurística **Greedy** (vecino más cercano) y la metaheurística **Variable Neighborhood Search (VNS)**, sobre las instancias públicas del NEO Research Group.

### Objetivos del proyecto

- Modelar el PVRP como un Proceso de Decisión de Markov (MDP).
- Implementar un entorno de simulación compatible con el estándar Gymnasium.
- Entrenar un agente de RL con Action Masking para respetar las restricciones de capacidad y patrones de visita.
- Comparar cuantitativamente el desempeño del agente contra Greedy y VNS sobre instancias del NEO Research Group.
- Evaluar la capacidad de generalización del agente (cero-shot y entrenamiento multi-instancia) sobre instancias no vistas.

---

## Estructura del repositorio

```
PVRP-DataDriven/
│
├── data/
│   ├── raw/                  # Instancias (.txt) y soluciones de referencia (.res) del NEO Research Group
│   └── processed/            # Reservado para instancias parseadas/serializadas (vacío por ahora)
│
├── src/                      # Código fuente del proyecto
│   ├── data/                 # Carga y representación de instancias y soluciones (Instance, Solution)
│   ├── environment/          # Entorno PVRP (MDP): estado, máscara de acciones, recompensa, PVRPEnv
│   ├── agent/                # Agente de RL: entrenamiento, evaluación, multi-semilla, configuración
│   ├── baselines/            # Métodos de referencia: Greedy y VNS (operadores + shaking)
│   └── utils/                # Utilidades transversales: distancias, soluciones, visualización
│
├── scripts/                  # CLIs ejecutables para correr entrenamiento, evaluación y experimentos
│
├── experiments/
│   └── configs/               # Plantillas de configuración (aún no consumidas por los scripts)
│
├── results/                  # Outputs generados (modelos, logs de TensorBoard, figuras) — no versionar
│
├── notebooks/                # Jupyter notebooks de exploración y análisis
│
└── tests/                    # Suite de tests (pytest)
```

### Descripción por módulo

| Módulo | Propósito |
|---|---|
| `src/data/` | Parsea archivos del NEO Research Group (`instance_loader.py`, `solution_loader.py`) y define las estructuras `Instance`, `Customer`, `Depot`. |
| `src/environment/` | Implementa `PVRPEnv` siguiendo la API de Gymnasium. Cada componente del MDP (`state.py`, `action_mask.py`, `reward.py`) vive en su propio archivo, más `multi_instance_env.py` para entrenamiento sobre varias instancias a la vez. |
| `src/agent/` | Entrenamiento del agente con `MaskablePPO` (`train.py`), evaluación determinística/estocástica y comparación contra baselines (`evaluate.py`), análisis multi-semilla (`multi_seed.py`) e hiperparámetros (`policy_config.py`). |
| `src/baselines/` | Implementaciones independientes de Greedy (`greedy.py`) y VNS (`vns.py`, `vns_operators.py`, `vns_shaking.py`), usadas como métodos de comparación. |
| `src/utils/` | Funciones transversales: distancias euclidianas, representación y validación de soluciones (`solution.py`), visualización de instancias y rutas. |
| `scripts/` | Punto de entrada real del proyecto: CLIs para validar el dataset, correr baselines, entrenar/evaluar el agente, y los experimentos puntuales citados en la memoria (ver [Uso](#uso)). |

---

## Requisitos

- Python ≥ 3.10
- Sistema operativo: Linux, macOS o Windows

### Dependencias principales

- `gymnasium` — Framework para entornos de RL.
- `sb3-contrib` — Implementación de MaskablePPO.
- `stable-baselines3`, `torch` — Base de algoritmos de RL.
- `numpy`, `pandas` — Manipulación de datos.
- `matplotlib`, `networkx` — Visualización de rutas y resultados.
- `tensorboard` — Monitoreo del entrenamiento.
- `pytest` — Tests unitarios.

Todas las dependencias están listadas en `requirements.txt`.

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/<usuario>/pvrp-rl-memoria.git
cd pvrp-rl-memoria

# Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# Instalar dependencias
pip install -r requirements.txt
```

---

## Uso

Todos los comandos se ejecutan desde la raíz del repositorio. Cada script agrega automáticamente el proyecto al `PYTHONPATH`, así que no es necesario instalarlo como paquete para probarlo.

### 1. Validar el dataset

Recorre todas las instancias en `data/raw/`, verifica que carguen correctamente y que su solución de referencia (BKS) sea factible y consistente:

```bash
python scripts/validate_dataset.py
python scripts/validate_dataset.py --export-csv results/dataset_summary.csv
```

### 2. Ejecutar los baselines (Greedy y VNS)

```bash
python scripts/baselines_all.py --instances p01 p02 p03 --vns-iters 100
```

### 3. Entrenar el agente de RL

```bash
python scripts/train_agent.py --instance p01 --timesteps 200000 --tensorboard
tensorboard --logdir results/tensorboard   # en otra terminal, para monitorear en vivo
```

El modelo entrenado se guarda en `results/models/ppo_<instance>.zip`.

### 4. Evaluar el agente y compararlo contra los baselines

```bash
python scripts/evaluate_agent.py --instance p01 --stochastic-runs 30 --vns-iters 150
```

### 5. Análisis multi-semilla

Entrena el agente con varias semillas sobre la misma instancia y reporta el gap medio ± desviación estándar:

```bash
python scripts/multi_seed.py --instance p01 --seeds 0 1 2 3 4 --timesteps 300000 --save-models
```

### 6. Generalización (cero-shot y multi-instancia)

```bash
# Evaluar un agente ya entrenado en instancias que nunca vio
python scripts/generalization.py zero-shot --train-instance p01 --test-instances p02 p03

# Entrenar un agente rotando entre varias instancias compatibles
python scripts/generalization.py multi --instances p01 p02 p03 --timesteps 300000 --tensorboard

# Evaluación cruzada de los modelos ya guardados en results/models/
python scripts/evaluate_cross.py
```

### 7. Visualizar rutas

```bash
python scripts/plot_routes.py --instance p01 --model results/models/ppo_p01 --compare-bks
```

Genera figuras PNG en `results/figures/`.

### Experimentos puntuales de la memoria

Los scripts `experimento_lambda_cero.py`, `experimento_lambda_barrido.py`, `experimento_lambda_p04.py`, `experimento_calibracion_p04.py`, `verificar_multi_seed_ganadora.py` y `ejemplo_mascara.py` reproducen análisis específicos citados en la memoria (reward hacking, calibración de hiperparámetros, ejemplo de Action Masking). No forman parte del pipeline principal y no reciben argumentos por línea de comandos; sus parámetros se ajustan editando el propio archivo.

---

## Tests

El proyecto usa `pytest` (configurado en `pyproject.toml`) con más de 140 tests que cubren carga de instancias/soluciones, el entorno PVRP, Action Masking, Greedy, VNS y la mecánica de entrenamiento/evaluación del agente:

```bash
pytest                      # corre toda la suite
pytest tests/test_greedy.py -v   # un módulo puntual
```

---

## Métricas de evaluación

- **Costo total de la ruta:** distancia euclidiana acumulada sobre el horizonte completo.
- **Gap respecto al mejor resultado conocido (BKS):** desviación porcentual frente a las soluciones reportadas por el NEO Research Group.
- **Tiempo de ejecución:** comparación de eficiencia computacional entre métodos.
- **Factibilidad:** verificación del cumplimiento de restricciones (capacidad vehicular, patrones de visita, frecuencias).
