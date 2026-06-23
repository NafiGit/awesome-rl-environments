# Awesome RL Environments [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of reinforcement learning environments — from classic control to the **LLM-agent training environments** powering today's frontier models.

Environments are the new bottleneck. As RL has moved from games to language agents, the limiting factor is no longer the algorithm but the **environment**: a sandbox that gives an agent a task, lets it act, and returns a reward. This list collects the best of them — battle-tested classics, modern embodied simulators, and the new wave of web, coding, and tool-use environments built specifically for training and evaluating LLM agents.

Contributions welcome — see [Contributing](#contributing).

## Contents

- [Standards & Infrastructure](#standards--infrastructure)
- [General-Purpose Frameworks](#general-purpose-frameworks)
- [LLM & Agent Environments](#llm--agent-environments)
  - [Web & GUI Agents](#web--gui-agents)
  - [Computer & OS Use](#computer--os-use)
  - [Coding & Software Engineering](#coding--software-engineering)
  - [Tool Use & Function Calling](#tool-use--function-calling)
  - [Reasoning, Text & Game Agents](#reasoning-text--game-agents)
- [Embodied AI & Robotics](#embodied-ai--robotics)
- [Games & Arcade](#games--arcade)
- [Multi-Agent](#multi-agent)
- [Board, Card & Strategy Games](#board-card--strategy-games)
- [Autonomous Driving](#autonomous-driving)
- [Domain-Specific](#domain-specific)
- [JAX-Accelerated & High-Throughput](#jax-accelerated--high-throughput)
- [Building Your Own](#building-your-own)
- [Related Awesome Lists](#related-awesome-lists)
- [Contributing](#contributing)
- [License](#license)

## Standards & Infrastructure

The APIs and runtimes everything else builds on.

- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) — The maintained fork of OpenAI Gym and the de-facto standard single-agent RL API.
- [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) — The Gymnasium of multi-agent RL; a standard API for environments with many interacting agents.
- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — An open standard and hub from Meta/PyTorch for agentic environments, designed for LLM-agent RL at scale.
- [EnvPool](https://github.com/sail-sg/envpool) — A C++-based, highly parallel environment executor; runs Atari/MuJoCo thousands of steps faster than vanilla Python.
- [Shimmy](https://github.com/Farama-Foundation/Shimmy) — Compatibility layer that wraps legacy and third-party environments (DM Control, OpenSpiel, etc.) into the Gymnasium API.
- [Minari](https://github.com/Farama-Foundation/Minari) — A standard format and hosting for offline-RL / imitation datasets collected from environments.

## General-Purpose Frameworks

- [DeepMind Control Suite (dm_control)](https://github.com/google-deepmind/dm_control) — Continuous-control benchmark tasks built on MuJoCo; a longtime standard for control research.
- [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) — Build environments in the Unity game engine and train agents with built-in RL algorithms.
- [Griddly](https://github.com/Bam4d/Griddly) — A high-performance engine for building grid-world games with rich observation and rendering options.
- [MiniGrid](https://github.com/Farama-Foundation/Minigrid) — Lightweight, fast grid-world environments for goal-directed and instruction-following research.
- [Gym-Retro / Stable-Retro](https://github.com/Farama-Foundation/stable-retro) — Turns classic console games into RL environments via emulators.

## LLM & Agent Environments

The fast-moving frontier: environments designed to train and evaluate language-model agents that browse, code, use tools, and operate computers.

- [Meta Agents Research Environments (ARE)](https://github.com/facebookresearch/meta-agents-research-environments) — A platform for building and running complex, tool-rich environments to study LLM agents.
- [AgentGym](https://github.com/WooooDyy/AgentGym) — A framework spanning many interactive environments for evolving and training generally-capable LLM agents.
- [AgentBench](https://github.com/THUDM/AgentBench) — A multi-environment benchmark evaluating LLMs as agents across OS, DB, web, and games.
- [BALROG](https://github.com/balrog-ai/BALROG) — A benchmark of long-horizon game environments (NetHack, MiniHack, BabyAI, Crafter) for agentic LLM/VLM reasoning.

### Web & GUI Agents

- [WebArena](https://github.com/web-arena-x/webarena) — A realistic, self-hostable web environment (shopping, forums, CMS, GitLab) for autonomous web agents.
- [VisualWebArena](https://github.com/web-arena-x/visualwebarena) — Extends WebArena with visually-grounded tasks that require understanding page screenshots.
- [BrowserGym](https://github.com/ServiceNow/BrowserGym) — A Gym-style environment for web agents, unifying WebArena, MiniWoB, WorkArena and more behind one API.
- [WebShop](https://github.com/princeton-nlp/WebShop) — A simulated e-commerce site with 1M+ products for training instruction-following shopping agents.
- [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web) — A dataset/environment for generalist web agents acting across hundreds of real websites.
- [WorkArena](https://github.com/ServiceNow/WorkArena) — Enterprise knowledge-work tasks on the ServiceNow platform, for agents doing real office workflows.

### Computer & OS Use

- [OSWorld](https://github.com/xlang-ai/OSWorld) — A real computer environment (Ubuntu/Windows/macOS) benchmarking agents on open-ended desktop tasks across apps.
- [AndroidWorld](https://github.com/google-research/android_world) — A live Android environment with 100+ hand-built tasks across 20 apps for mobile-device agents.
- [AndroidEnv](https://github.com/google-deepmind/android_env) — DeepMind's RL environment exposing the Android OS as a touchscreen control problem.
- [AppWorld](https://github.com/StonyBrookNLP/appworld) — A simulated world of 9 apps and 457 APIs for benchmarking interactive coding/tool agents.
- [Terminal-Bench](https://github.com/laude-institute/terminal-bench) — A benchmark + environment for agents that accomplish real tasks in a command-line terminal.

### Coding & Software Engineering

- [SWE-bench](https://github.com/princeton-nlp/SWE-bench) — Resolve real GitHub issues in real repositories; the standard benchmark for software-engineering agents.
- [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) — The first training environment for SWE agents, with executable repos and reward via the test suite.
- [debug-gym](https://github.com/microsoft/debug-gym) — A text environment from Microsoft for agents that interactively debug code using tools like a Python debugger.
- [MLE-bench](https://github.com/openai/mle-bench) — OpenAI's benchmark of 75 Kaggle competitions measuring how well agents do end-to-end ML engineering.

### Tool Use & Function Calling

- [τ-bench (tau-bench)](https://github.com/sierra-research/tau-bench) — Evaluates agents on tool-use and policy-following in dynamic, user-in-the-loop customer-service domains.
- [τ²-bench (tau2-bench)](https://github.com/sierra-research/tau2-bench) — A dual-control successor where both the agent and a simulated user can act on the environment.
- [ToolBench](https://github.com/OpenBMB/ToolBench) — A large-scale environment/dataset for training and evaluating tool-use over 16k+ real APIs.
- [Gorilla / APIBench](https://github.com/ShishirPatil/gorilla) — Connect LLMs to thousands of APIs; a reference environment for function-calling research.

### Reasoning, Text & Game Agents

- [Verifiers](https://github.com/willccbb/verifiers) — A library for writing RL environments for LLMs with verifiable rewards; popular for GRPO-style training.
- [TextArena](https://github.com/LeonGuertler/TextArena) — 100+ competitive and cooperative text games for training and evaluating LLMs through self-play.
- [Reasoning Gym](https://github.com/open-thought/reasoning-gym) — Procedurally-generated reasoning tasks with verifiable answers, built for RL with verifiable rewards.
- [TextWorld](https://github.com/microsoft/TextWorld) — Microsoft's engine for generating text-adventure games as language-grounded RL environments.
- [ALFWorld](https://github.com/alfworld/alfworld) — Aligns TextWorld tasks with embodied ALFRED tasks, letting agents learn abstractly then act.
- [ScienceWorld](https://github.com/allenai/ScienceWorld) — A text environment testing whether agents can perform grade-school science experiments and reasoning.

## Embodied AI & Robotics

- [MuJoCo](https://github.com/google-deepmind/mujoco) — DeepMind's fast, accurate physics engine underpinning much of continuous-control RL.
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) — NVIDIA's GPU-accelerated framework for robot learning, supporting thousands of parallel simulated environments.
- [Habitat](https://github.com/facebookresearch/habitat-lab) — A high-performance simulator for embodied agents navigating photorealistic 3D indoor scenes.
- [ManiSkill](https://github.com/haosulab/ManiSkill) — GPU-parallelized robotic manipulation environments with thousands of objects.
- [robosuite](https://github.com/ARISE-Initiative/robosuite) — A modular MuJoCo-based simulation framework for robot manipulation benchmarks.
- [Meta-World](https://github.com/Farama-Foundation/Metaworld) — 50 robotic manipulation tasks designed for meta-learning and multi-task RL.
- [Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics) — A collection of robotics environments (Fetch, Shadow Hand, maze) under the Gymnasium API.

## Games & Arcade

- [Arcade Learning Environment (ALE)](https://github.com/Farama-Foundation/Arcade-Learning-Environment) — The Atari 2600 benchmark that launched deep RL; still a standard for general competence.
- [Procgen Benchmark](https://github.com/openai/procgen) — 16 procedurally-generated arcade games for measuring generalization, not memorization.
- [NetHack Learning Environment (NLE)](https://github.com/heiner/nle) — The famously hard roguelike NetHack as a fast RL environment for long-horizon exploration.
- [MiniHack](https://github.com/facebookresearch/minihack) — A sandbox built on NLE for designing custom, controllable roguelike RL tasks.
- [Crafter](https://github.com/danijar/crafter) — An open-world survival game that evaluates a broad spectrum of agent abilities in one benchmark.
- [Craftax](https://github.com/MichaelTMatthews/Craftax) — A JAX reimplementation of Crafter that runs orders of magnitude faster for open-ended RL.
- [MineDojo](https://github.com/MineDojo/MineDojo) — A massively multitask Minecraft environment with thousands of tasks and an internet-scale knowledge base.
- [MineRL](https://github.com/minerllabs/minerl) — Minecraft environments paired with large human-demonstration datasets for imitation + RL.
- [Voyager](https://github.com/MineDojo/Voyager) — An open-ended, LLM-powered lifelong-learning agent for Minecraft; a reference for agent-in-environment design.

## Multi-Agent

- [SMACv2](https://github.com/oxwhirl/smacv2) — The StarCraft Multi-Agent Challenge; the standard cooperative MARL micromanagement benchmark.
- [Melting Pot](https://github.com/google-deepmind/meltingpot) — DeepMind's suite for evaluating multi-agent generalization to novel social situations.
- [Google Research Football](https://github.com/google-research/football) — A physics-based 11-vs-11 football environment for cooperative and competitive RL.
- [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai) — A cooperative cooking game widely used to study human-AI coordination.
- [Neural MMO](https://github.com/NeuralMMO/environment) — A massively-multi-agent environment simulating many agents competing for resources in a persistent world.
- [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) — A vectorized 2D physics simulator for fast multi-robot / swarm MARL.

## Board, Card & Strategy Games

- [OpenSpiel](https://github.com/google-deepmind/open_spiel) — DeepMind's collection of 70+ games and algorithms for research in RL and search/planning.
- [PySC2](https://github.com/google-deepmind/pysc2) — DeepMind's StarCraft II Learning Environment, exposing the full game as an RL challenge.
- [RLCard](https://github.com/datamllab/rlcard) — A toolkit of card games (poker, blackjack, mahjong, UNO) for RL with imperfect information.
- [Pgx](https://github.com/sotetsuk/pgx) — Vectorized, JAX-native board games (chess, Go, shogi, backgammon) for high-throughput self-play.

## Autonomous Driving

- [CARLA](https://github.com/carla-simulator/carla) — An open-source simulator for autonomous-driving research with sensors, maps, and traffic.
- [MetaDrive](https://github.com/metadriverse/metadrive) — A lightweight, procedurally-generated driving simulator for generalizable RL.
- [highway-env](https://github.com/Farama-Foundation/HighwayEnv) — Minimalist 2D autonomous-driving and tactical-decision environments under the Gymnasium API.

## Domain-Specific

- [FinRL](https://github.com/AI4Finance-Foundation/FinRL) — A framework of market environments for reinforcement learning in quantitative trading.
- [Flatland](https://github.com/flatland-association/flatland-rl) — A train-scheduling environment for multi-agent path-finding on railway networks.
- [Gym-ANM](https://github.com/robinhenry/gym-anm) — Environments for RL control of electricity distribution / active network management.
- [CityLearn](https://github.com/intelligent-environments-lab/CityLearn) — A benchmark for RL on building energy coordination and demand response.

## JAX-Accelerated & High-Throughput

End-to-end GPU/TPU environments that remove the CPU-simulation bottleneck.

- [Brax](https://github.com/google/brax) — A differentiable, massively-parallel rigid-body physics engine for continuous control in JAX.
- [Gymnax](https://github.com/RobertTLange/gymnax) — JAX reimplementations of classic-control, bsuite, and MinAtar environments for fully on-accelerator RL.
- [Jumanji](https://github.com/instadeep/jumanji) — A suite of scalable, JAX-based combinatorial and routing environments from InstaDeep.
- [JaxMARL](https://github.com/FLAIROx/JaxMARL) — Multi-agent environments in JAX (incl. SMAX, Overcooked, MPE) for orders-of-magnitude faster MARL.
- [PGX](https://github.com/sotetsuk/pgx) — JAX-native board games for high-throughput self-play (also listed above).

## Building Your Own

- [Gymnasium — Creating environments](https://gymnasium.farama.org/introduction/create_custom_env/) — The official guide to wrapping any task in the standard RL API.
- [PettingZoo — Environment creation](https://pettingzoo.farama.org/content/environment_creation/) — How to author a custom multi-agent environment.
- [Verifiers — environment guide](https://github.com/willccbb/verifiers) — Patterns for writing verifiable-reward environments for LLM RL.
- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — A spec for packaging agentic environments so they are portable across training stacks.

## Related Awesome Lists

- [Awesome Reinforcement Learning](https://github.com/aikorea/awesome-rl) — Broad RL resources: papers, courses, and libraries.
- [Awesome Deep RL](https://github.com/kengz/awesome-deep-rl) — Deep-RL frameworks, algorithms, and applications.
- [Awesome LLM-Powered Agent](https://github.com/hyp1231/awesome-llm-powered-agent) — Papers and resources on LLM agents (complements the environments here).

## Contributing

Contributions are very welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first. In short: one environment per entry, keep descriptions to a single neutral sentence, link to the canonical source, and add it to the most fitting section (alphabetical-ish within a section). Suggestions, corrections, and new categories are all appreciated — open an issue or a PR.

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](LICENSE)

To the extent possible under law, the contributors have waived all copyright and related or neighboring rights to this work. See [LICENSE](LICENSE).
