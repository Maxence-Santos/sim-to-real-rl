# Roadmap : Projet Sim-to-Real Quadrupède

Ce document détaille les différentes phases de réalisation du projet.

## Phase 1 — Fondations RL & Simulation (Semaines 1-2)
**Objectif :** Mettre en place l'environnement de base et valider le pipeline RL.
- [x] Installer MuJoCo et le wheel CPU de PyTorch (`./scripts/install_cpu.sh`). Isaac Lab est hors v1.
- [ ] Réviser les concepts RL (MDP, policy gradient, PPO).
- [x] Vérifier le pipeline PPO directement sur l'environnement A1 (pas de cart-pole intermédiaire).
- [x] Se familiariser avec Stable Baselines3.
- **Milestone :** le pipeline PPO tourne sur CPU, et une marche MuJoCo sans randomization est entraînée.

## Phase 2 — Environnement Quadrupède (Semaines 3-4)
**Objectif :** Développer l'environnement de simulation pour le robot quadrupède.
- [x] Charger un modèle de quadrupède (Unitree A1, MuJoCo Menagerie).
- [x] Définir l'observation space (48 valeurs : joints, vitesses du tronc, gravité, commande, action précédente).
- [x] Définir l'action space (résidus de position des 12 articulations).
- [x] Implémenter les reward functions (velocity tracking, energy penalty, stability).
- [x] Entraîner une première politique de marche basique (`models/walk_baseline.zip`, 10 M pas, sans DR).
- **Milestone :** atteint. Sur 8 épisodes d'évaluation : 20 s sans chute, erreur de vitesse 0,016 m/s.

## Phase 3 — Domain Randomization (Semaines 5-6)
**Objectif :** Rendre la politique robuste aux variations physiques pour préparer le transfert.
- [x] Implémenter la randomisation des paramètres physiques :
  - Masse du robot.
  - Friction du sol.
  - Latence des actions.
  - Bruit des capteurs.
- [x] Entraîner une nouvelle politique avec le Domain Randomization (DR) activé (`models/walk_dr.zip`, 10 M pas, 4056 s).
- [x] Comparer les performances (baseline sans DR vs avec DR) sur 16 épisodes appariés, graines 1 à 16.
- [x] Analyser l'impact de chaque paramètre de DR. Les récompenses ne sont pas modifiées : le trou mesuré est le délai d'action nul, absent de l'entraînement.
- **Milestone :** sous la randomization complète, la politique DR tient 20 s, erreur 0,092 m/s, 16/16. Le baseline est à 0,399 m/s et 5/16. Sans le délai vu à l'entraînement (1 ou 2 pas), la politique DR marche à environ 0,16 m/s. Détail dans `docs/dr_comparison.md`.

## Phase 4 — Transfert Sim-to-Sim (Semaines 7-8)
**Objectif :** Transférer la politique entraînée vers un simulateur différent (Gazebo) pour simuler un transfert vers le monde réel.
- [x] Exporter la politique PyTorch vers ONNX (acteur déterministe, CPU). `walk_baseline.onnx` et `walk_dr.onnx` coïncident avec le module PyTorch à environ 1e-5.
- [x] Créer le bridge Gazebo. Le nœud ROS 2 `a1_policy_bridge` publie `/a1/joint_states` et `/a1/odom`. Les consignes articulaires partent en gz-transport.
- [x] Tester le transfert zero-shot (MuJoCo → Gazebo), un épisode déterministe de 20 s par commande (0,40 / 0,55 / 0,70 m/s).
- [x] Mesurer le *performance gap*. Aucune chute. À 0,70 m/s le baseline Gazebo est à 0,21 m/s (erreur 0,508) contre 0,70 m/s dans MuJoCo.
- [x] Comparer le transfert de la politique avec DR vs sans DR. `walk_dr` reste à 0,64 m/s sur cette commande (erreur 0,078), proche de son MuJoCo (0,64 m/s).
- **Milestone :** atteint pour le sim-to-sim. Le détail chiffré est dans `docs/gazebo_transfer.md`. Pas de robot physique.

## Phase 5 — Analyse & Polish (Semaines 9-10)
**Objectif :** Finaliser le projet pour le rendre présentable et professionnel.
- [x] Générer les courbes d'entraînement depuis TensorBoard (PPO_4 sans DR, retour final ~3172 ; PPO_5 avec DR, ~1580 puis ~3093). Pas de WandB. `docs/figures/training_return.png`.
- [x] Documenter les résultats. Tableaux dans `docs/dr_comparison.md` et `docs/gazebo_transfer.md`. Graphiques `docs/figures/dr_with_latency.png` et `docs/figures/gazebo_speed_error.png`.
- [x] Enregistrer une démo Sim vs Gazebo. GIF 8 s de `walk_dr` à 0,70 m/s : MuJoCo 0,62 m/s, Gazebo 0,53 m/s, aucune chute. Caméra Gazebo de niveau et damier visuel. `docs/figures/sim_vs_gazebo.gif`. Prise séparée du tableau de 20 s.
- [x] Finaliser le README (anglais), avec les courbes et le GIF.
- **Milestone :** courbes, GIF et README sont en place pour le sim-to-sim. Pas de robot physique. Le projet suivant du portfolio n'est pas commencé.
