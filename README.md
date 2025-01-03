# LLM Dynamic Planner (LLM-DP)

LLM Dynamic Planner (LLM-DP) is a framework that integrates Large Language Models (LLMs) with automated planning systems to solve complex tasks in dynamic environments. It leverages the reasoning capabilities of LLMs to generate high-level plans and uses a symbolic planner (e.g., LAPKT) to refine and execute these plans. This approach is particularly useful in environments like ALFWorld and TextWorld, where agents must interact with a simulated world to achieve specific goals.

## Goals

- **Combine LLMs and Symbolic Planning**: Bridge the gap between high-level reasoning (LLMs) and low-level task execution (symbolic planners) to solve complex tasks efficiently.
- **Dynamic Adaptation**: Enable the system to adapt to changes in the environment by dynamically updating plans based on real-time feedback.
- **Generalization**: Improve generalization across tasks by leveraging the knowledge encoded in LLMs and the robustness of symbolic planners.
- **Benchmarking**: Provide a framework for benchmarking the performance of LLM-based planning systems in interactive environments like ALFWorld and TextWorld.

## Key Features

- **Integration with ALFWorld and TextWorld**: Seamlessly integrates with both ALFWorld and TextWorld environments for task-oriented simulations.
- **Support for Multiple Planners**: Supports both BFS-F and FF planners for generating and executing plans.
- **Customizable Configurations**: Allows users to customize various parameters such as timeouts, CPU usage, and LLM models.
- **ReAct Baseline**: Includes an implementation of the ReAct baseline for comparison with LLM-DP.
- **Random Fallback**: Provides a random fallback mechanism for handling cases where the planner or LLM fails to generate a valid plan.

## Setup

1. Install ``alfworld`` following instructions [here](https://github.com/alfworld/alfworld).
2. Install ``textworld`` using pip: ``pip install textworld``.
3. Install requirements: ``pip install -r requirements.txt``.
4. Install docker for running LAPKT planner. There are two different docker images available:
   - The docker image for the linux/arm64 platform is available [here](<https://hub.docker.com/repository/docker/gautierdag/lapkt-arm/general>). See the `Dockerfile` for more details.
   - The docker image for the linux/amd64 platform is available [here](<https://hub.docker.com/r/lapkt/lapkt-public>).

## Config

The config file is a ``.env`` file. The following variables are available:

- ``openai_api_key``: OpenAI API key.
- ``alfworld_data_path``: path to the ``alfworld/data`` directory.
- ``alfworld_config_path``: path to the ``alfworld/configs`` directory.
- ``pddl_dir``: path to the directory where PDDL files are stored.
- ``pddl_domain_file``: name of the PDDL domain file.
- ``planner_solver``: name of the planner solver to use. Currently, only ``bfs_f`` and ``ff`` are supported.
- ``planner_timeout``: timeout for the planner in seconds (default: 30).
- ``planner_cpu_count``: number of CPUs to use for the planner (default: 4).
- ``top_n``: number of plans to generate (default: 3).
- ``platform``: platform to use for the planner (default: ``linux/arm64``).
- ``output_dir``: path to the output directory (default: ``output``).
- ``seed``: random seed (default: 42).
- ``name``: name of the experiment (default: ``llmdp``).
- ``llm_model``: name of the LLM model to use (default: ``gpt-3.5-turbo-0613``).
- ``sample``: whether to use LLM to instantiate beliefs (default: ``llm``).
- ``use_react_chat``: activate ReAct baseline (default: ``False``).
- ``random_fallback``: activate random fallback (default: ``False``).

Copy the ``.env.draft`` to ``.env`` and fill the variables in the created ``.env`` file with the appropriate values.

See `config.py` for more details.

## Run

Run the following command to run the LLM-DP (or ReAct) agent:

```bash
python main.py
