import re
import random
from typing import Literal
from collections import defaultdict, Counter
import textworld
from utils.planner import parallel_lapkt_solver
from utils.llm_utils import llm_cache

# Few-shot prompt for generating PDDL goals
GENERATE_GOAL_PROMPT = [
    {
        "role": "system",
        "content": """(define (domain textworld)
(:predicates
(isReceptacle ?o - object) ; true if the object is a receptacle
(atReceptacleLocation ?r - object) ; true if the agent is at the receptacle location
(inReceptacle ?o - object ?r - object) ; true if object ?o is in receptacle ?r
(holds ?o - object) ; object ?o is held by agent
))""",
    },
    {
        "role": "user",
        "content": "Your task is to: take the key and unlock the door.",
    },
    {
        "role": "assistant",
        "content": """(:goal
(exists (?t - key ?d - door)
(and (holds ?t)
)))""",
    },
    {
        "role": "user",
        "content": "Your task is to: find the treasure and take it.",
    },
    {
        "role": "assistant",
        "content": """(:goal
(exists (?t - treasure)
(and (holds ?t)
)))""",
    },
]

class LLMDPAgent:
    """
    LLM-DP agent adapted for TextWorld.
    """

    def __init__(
        self,
        initial_scene_observation: str,
        task_description: str,
        logger=None,
        sample: Literal["llm", "random"] = "llm",
        top_n=3,
        random_fallback=False,
        temperature=0.0,
    ) -> None:
        self.initial_scene_observation = initial_scene_observation
        self.task_description = task_description
        self.llm_tokens_used = 0
        self.logger = logger
        self.sample = sample
        self.top_n = top_n
        self.random_fallback = random_fallback
        self.temperature = temperature

        # Initialize scene objects and beliefs
        self.scene_objects = defaultdict(dict)
        self.initialize_scene_objects(initial_scene_observation)

        # Generate PDDL goal from task description
        self.pddl_goal = self.get_pddl_goal()
        self.logger.info(f"PDDL GOAL: {self.pddl_goal}")

        # Initialize actions taken
        self.actions_taken = []

    def initialize_scene_objects(self, observation: str) -> None:
        """
        Initialize scene objects and beliefs based on the initial observation.
        """
        # Extract objects from the observation
        objects = re.findall(r"a (\w+)", observation)
        for obj in objects:
            self.scene_objects[obj] = {"type": obj, "beliefs": {"inReceptacle": []}}

    def process_obs(self, observation: str) -> dict:
        """
        Process TextWorld observations into a structured format.
        """
        json_dict = {}
        # Extract objects and their locations
        objects = re.findall(r"a (\w+)", observation)
        for obj in objects:
            json_dict[obj] = {"location": "unknown"}  # Update based on observation
        return json_dict

    def get_pddl_goal(self) -> str:
        """
        Generate a PDDL goal from the task description using an LLM.
        """
        prompt_messages = GENERATE_GOAL_PROMPT + [
            {
                "role": "user",
                "content": self.task_description,
            }
        ]
        pddl_goal, token_usage = llm_cache(
            prompt_messages, stop=None, temperature=self.temperature
        )
        self.llm_tokens_used += token_usage["total_tokens"]
        return pddl_goal

    def get_pddl_belief_predicate(
        self, init_str: str, belief_predicate: str, belief_values: list[str], top_n: int
    ) -> list[str]:
        """
        Use the LLM to predict the most likely values for an unknown predicate.
        """
        user_prompt = (
            f"Predict: {belief_predicate}\n"
            + f"Select the top {top_n} likely items for ? from the list:"
            + f"{sorted(belief_values)}\n"
            + "Return a parsable python list of choices."
        )
        prompt_messages = [
            {"role": "system", "content": f"Observed Environment\n{init_str}"},
            {"role": "user", "content": user_prompt},
        ]
        selected_values, token_usage = llm_cache(
            prompt_messages, stop=None, temperature=self.temperature
        )
        self.llm_tokens_used += token_usage["total_tokens"]
        try:
            selected_values = re.findall(r"'(.*?)'", selected_values)
        except Exception as e:
            self.logger.info(f"Error parsing selected values: {selected_values}")
            raise e
        return selected_values

    def get_pddl_objects(self) -> str:
        """
        Generate the PDDL objects section.
        """
        objects_str = "".join(
            [f"{o} - {atts['type']}\n" for o, atts in self.scene_objects.items()]
        )
        return f"(:objects {objects_str})\n"

    def get_pddl_init(self, sample="random") -> list[str]:
        """
        Generate the PDDL initial state section.
        """
        known_predicates = ""
        for r, atts in self.scene_objects.items():
            for att, val in atts.items():
                if att in ["type", "beliefs"] or val is False:
                    continue
                if val is True:
                    known_predicates += f"({att} {r})\n"
                else:
                    known_predicates += f"({att} {r} {val})\n"

        belief_predicates = [known_predicates] * self.top_n
        for o, atts in self.scene_objects.items():
            if "beliefs" in atts:
                for belief_attribute in atts["beliefs"]:
                    options = atts["beliefs"][belief_attribute]
                    if sample == "random":
                        sampled_beliefs = random.choices(options, k=self.top_n)
                    elif sample == "llm":
                        sampled_beliefs = self.get_pddl_belief_predicate(
                            init_str=known_predicates,
                            belief_predicate=f"({belief_attribute} {o} ?)",
                            belief_values=options,
                            top_n=self.top_n,
                        )
                    else:
                        raise ValueError(f"Unknown sample method: {sample}")

                    belief_predicates = list(
                        map(
                            lambda x, s: x + f"({belief_attribute} {o} {s})\n",
                            belief_predicates,
                            sampled_beliefs,
                        )
                    )

        return list(set([f"(:init {predicates})\n" for predicates in belief_predicates]))

    def get_pddl_problem(self, sample: Literal["llm", "random"] = "llm") -> list[str]:
        """
        Generate PDDL problem files.
        """
        inits = self.get_pddl_init(sample=sample)
        problems = []
        for init in inits:
            problems.append(
                "(define (problem textworld)\n(:domain textworld)\n"
                + f"{self.get_pddl_objects()}{init}{self.pddl_goal})"
            )
        return problems

    def update_observation(self, observation: str) -> bool:
        """
        Update the agent's beliefs based on the observation.
        """
        scene_obs = self.process_obs(observation)
        for obj, info in scene_obs.items():
            self.scene_objects[obj]["location"] = info["location"]
        return len(scene_obs) > 0

    def get_plan(self) -> list[str]:
        """
        Generate a plan using the symbolic planner.
        """
        problems = self.get_pddl_problem(sample=self.sample)
        plans = parallel_lapkt_solver(problems, logger=self.logger)
        if self.random_fallback and len(plans) == 0:
            self.logger.warning("No plans found: sampling randomly.")
            problems = self.get_pddl_problem(sample="random")
            plans = parallel_lapkt_solver(problems, logger=self.logger)
        return min(plans, key=len)

    def take_action(self, last_observation="") -> str:
        """
        Select and execute the next action.
        """
        if last_observation == "Nothing happens.":
            self.logger.warning("Invalid Action: No observation received.")
            return "look"  # Default action to observe the environment

        # Update beliefs based on the observation
        changed = self.update_observation(last_observation)
        if changed:
            self.plan = self.get_plan()

        # Get the next action from the plan
        pddl_action = self.plan.pop(0)
        action_args = pddl_action.split(" ")
        return self.convert_pddl_action_to_textworld(action_args[0], action_args[1:])

    def convert_pddl_action_to_textworld(self, action_name: str, action_args: list[str]) -> str:
        """
        Convert PDDL actions to TextWorld commands.
        """
        match action_name:
            case "gotoreceptacle":
                return f"go to {action_args[0]}"
            case "pickupobjectfromreceptacle":
                return f"take {action_args[0]}"
            case "putobject":
                return f"put {action_args[0]} in {action_args[1]}"
            case _:
                raise ValueError(f"Unknown action: {action_name}")


def main():
    # Generate a TextWorld game
    game = textworld.generator.make_game()
    env = textworld.start(game)

    # Initialize the LLM-DP agent
    initial_obs = env.reset()
    agent = LLMDPAgent(initial_obs, "Your task is to: solve the game.")

    # Main loop
    while not env.is_done():
        action = agent.take_action(env.get_observation())
        obs, reward, done = env.step(action)
        agent.update_observation(obs)


if __name__ == "__main__":
    main()
