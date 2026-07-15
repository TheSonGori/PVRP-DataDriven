# PVRP-RL: Resolución del Problema de Ruteo de Vehículos Periódico mediante Aprendizaje por Refuerzo

Repositorio asociado a la memoria de título **"Resolución del Problema de Ruteo de Vehículos Periódico (PVRP) mediante un Enfoque Data-Driven con Aprendizaje por Refuerzo"**.

**Autora:** Javiera Elena Gutiérrez Abarca

---

## Descripción

El **Periodic Vehicle Routing Problem (PVRP)** es una extensión del clásico Vehicle Routing Problem (VRP) en la que cada cliente debe ser visitado con una frecuencia determinada a lo largo de un horizonte de planificación de varios días. El problema integra dos decisiones acopladas: la asignación temporal de visitas (¿qué días visitar a cada cliente?) y la construcción espacial de rutas (¿en qué orden recorrerlos?). Su naturaleza combinatoria lo clasifica como **NP-hard**, lo que impide resolverlo de manera exacta para instancias de tamaño realista.

Este proyecto propone un enfoque **data-driven** basado en **Aprendizaje por Refuerzo (RL)** para abordar el PVRP. Un agente entrenado aprende, por interacción con un entorno simulado, a construir soluciones de manera secuencial, sin depender de reglas diseñadas manualmente. El desempeño del modelo se compara contra dos métodos de referencia ampliamente utilizados en la literatura: una heurística **Greedy** (vecino más cercano) y la metaheurística **Variable Neighborhood Search (VNS)**.

### Objetivos del proyecto

- Modelar el PVRP como un Proceso de Decisión de Markov (MDP).
- Implementar un entorno de simulación compatible con el estándar Gymnasium.
- Entrenar un agente de RL con Action Masking para respetar las restricciones de capacidad y patrones de visita.
- Comparar cuantitativamente el desempeño del agente contra Greedy y VNS sobre instancias del NEO Research Group.

---

## Estructura del repositorio

```
pvrp-rl-memoria/
│
├── data/
│   ├── raw/                  # Instancias originales del NEO Research Group (.txt)
│   └── processed/            # Instancias parseadas y serializadas
│
├── src/                      # Código fuente del proyecto
│   ├── data/                 # Carga y representación de instancias
│   ├── environment/          # Entorno PVRP (MDP): estado, acciones, recompensa, máscaras
│   ├── agent/                # Agente de RL: entrenamiento, evaluación, configuración
│   ├── baselines/            # Métodos de referencia: Greedy y VNS
│   └── utils/                # Utilidades: distancias, soluciones, métricas, visualización
│
├── experiments/              # Scripts ejecutables para correr experimentos
│   └── configs/              # Archivos de configuración (YAML/JSON)
│
├── results/                  # Outputs (modelos, logs, soluciones, figuras) — ignorado por Git
│
├── notebooks/                # Jupyter notebooks para exploración y análisis
│
├── tests/                    # Tests unitarios
│
└── docs/                     # Documentación complementaria
```

### Descripción por módulo

| Módulo | Propósito |
|---|---|
| `src/data/` | Parsea archivos del NEO Research Group y construye objetos `Instance` reutilizables. |
| `src/environment/` | Implementa el entorno `PVRPEnv` siguiendo la API de Gymnasium. Cada componente del MDP (estado, máscara de acciones, recompensa) vive en su propio archivo para facilitar mantenimiento y citación en la memoria. |
| `src/agent/` | Entrenamiento del agente con `MaskablePPO` (sb3-contrib), evaluación sobre instancias y configuración de hiperparámetros. |
| `src/baselines/` | Implementaciones independientes de Greedy y VNS, utilizadas como métodos de comparación. |
| `src/utils/` | Funciones transversales: cálculo de distancias euclidianas, representación de soluciones, métricas comparativas y visualización de rutas. |
| `experiments/` | Scripts que orquestan los módulos para ejecutar entrenamientos, evaluaciones y comparaciones. |

---

## Requisitos

- Python ≥ 3.10
- Sistema operativo: Linux, macOS o Windows

### Dependencias principales

- `gymnasium` — Framework para entornos de RL.
- `sb3-contrib` — Implementación de MaskablePPO.
- `stable-baselines3` — Base de algoritmos de RL.
- `numpy`, `pandas` — Manipulación de datos.
- `networkx`, `matplotlib` — Visualización de rutas y resultados.
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

### 1. Preparar instancias

Las instancias originales deben colocarse en `data/raw/`.

### 2. Ejecutar baselines

```bash
python experiments/run_baselines.py --instance pr01
```

### 3. Entrenar el agente de RL

```bash
python experiments/run_training.py --config experiments/configs/ppo_default.yaml
```

### 4. Comparar resultados

```bash
python experiments/run_comparison.py --instances pr01 pr02 pr03
```

Los resultados se almacenan en `results/` (modelos entrenados, logs de TensorBoard, soluciones en JSON y figuras finales).

---

## Métricas de evaluación

- **Costo total de la ruta:** distancia euclidiana acumulada sobre el horizonte completo.
- **Gap respecto al mejor resultado conocido (BKS):** desviación porcentual frente a las soluciones reportadas por el NEO Research Group.
- **Tiempo de ejecución:** comparación de eficiencia computacional entre métodos.
- **Factibilidad:** verificación del cumplimiento de restricciones (capacidad vehicular, patrones de visita, frecuencias).

---