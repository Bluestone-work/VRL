"""VRL V3: graph-based swarm configuration and interaction-aware navigation.

Modules
-------
assumptions  registered physical, resource and objective parameters (configs/v3/assumptions.json)
scene        base tasks independent of cluster number (base_task_hash) and deployment-aware environments
vessel_graph weighted vessel graph G = (V, E, W) built from the simulator's preoperative geometry
configuration joint optimisation of N, release points, allocation, routes and release/passage schedule
metrics      the four-metric evaluator used by every V3 experiment
execute      run a configuration in the simulator with a fixed local controller
"""
