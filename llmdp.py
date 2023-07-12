import json
import os
import pickle
import random
import re
import time
from collections import Counter, defaultdict

import openai
import yaml
from openai.error import RateLimitError

from config import LLMDPConfig
from logger import get_logger
from planner import parallel_lapkt_solver

openai.api_key = LLMDPConfig.openai_api_key


GENERATE_GOAL_PROMPT = [
    {
        "role": "system",
        "content": """(define (domain alfred)
(:predicates
(isReceptacle ?o - object) ; true if the object is a receptacle
(atReceptacleLocation ?r - object) ; true if the robot is at the receptacle location
(inReceptacle ?o - object ?r - object) ; true if object ?o is in receptacle ?r
(openable ?r - object) ; true if a receptacle is openable
(opened ?r - object) ; true if a receptacle is opened
(isLight ?o - object) ; true if an object is light source
(examined ?o - object ?l - object) ; whether the object has been looked at with light
(holds ?o - object) ; object ?o is held by robot
(isClean ?o - object) ; true if the object has been cleaned in sink
(isHot ?o - object) ; true if the object has been heated up
(isCool ?o - object) ; true if the object has been cooled
(isSink ?o - object) ; true if the object is a sink
(isMicrowave ?o - object) ; true if the object is a microwave
(isFridge ?o - object) ; true if the object is a fridge
))""",
    },
    {
        "role": "user",
        "content": "Your task is to: put a clean plate in microwave.",
    },
    {
        "role": "assistant",
        "content": """(:goal
(exists (?t - plate ?r - microwave)
(and (inReceptacle ?t ?r)
(isClean ?t)
)))""",
    },
    {
        "role": "user",
        "content": "Your task is to: examine an alarmclock with the desklamp",
    },
    {
        "role": "assistant",
        "content": """(:goal
(exists (?t - alarmclock ?l - desklamp)
(and (examined ?t ?l) (holds ?t)
)))""",
    },
    {"role": "user", "content": "Your task is to: put two cellphone in bed"},
    {
        "role": "assistant",
        "content": """(:goal
(exists (?t1 - cellphone ?t2 - cellphone ?r - bed)
(and (inReceptacle ?t1 ?r)
(inReceptacle ?t2 ?r)
(not (= ?t1 ?t2))
)))""",
    },
]


def llm(llm_messages: list[str], stop=None) -> tuple[str, dict[str, int]]:
    try:
        completion = openai.ChatCompletion.create(
            model=LLMDPConfig.llm_chat_model,
            messages=llm_messages,
            temperature=0.0,
            max_tokens=100,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
        )
        return completion.choices[0].message["content"], dict(completion["usage"])
    except RateLimitError:
        time.sleep(10)
        return llm(llm_messages, stop=stop)


def llm_cache(
    llm_messages: list[dict[str]],
    stop=None,
) -> tuple[str, int]:
    cache_file = (
        f"{LLMDPConfig.output_dir}/llm_responses_{LLMDPConfig.llm_chat_model}.pickle"
    )
    llm_responses = {}
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            llm_responses = pickle.load(f)

    key = json.dumps(llm_messages)
    if key not in llm_responses:
        generated_content, token_usage = llm(llm_messages, stop=stop)
        llm_responses[key] = (generated_content, token_usage)
        with open(cache_file, "wb") as f:
            pickle.dump(llm_responses, f)

    return llm_responses[key]


class LLMDPAgent:
    """
    Alfworld agent that uses the LLMDP planner to generate a plan to complete a task.
    """

    def __init__(
        self,
        initial_scene_observation: str,
        task_description: str,
        env_id=0,
        logger=None,
        use_llm_search=False,
        top_n=3,
    ) -> None:
        self.initial_scene_observation = initial_scene_observation
        self.task_description = task_description
        self.env_id = env_id
        self.llm_tokens_sent = 0
        self.llm_chat_tokens_sent = 0
        self.logger = logger
        self.use_llm_search = use_llm_search
        self.top_n = top_n

        # get PDDL objects from scene_observation
        self.scene_receptacles = self.find_receptacles_from_scene_observation(
            initial_scene_observation
        )

        # populate scene_objects with all objects/receptacles in scene
        self.scene_objects = defaultdict(dict)
        for r in self.scene_receptacles:
            self.scene_objects[r] = self.get_receptacle_attributes(r)

        self.pddl_goal = self.get_pddl_goal()
        self.logger.info(f"PDDL GOAL: {self.pddl_goal}")

        # Get objects that should exist based on generated PDDL goal
        existential_pattern = r"\?\w+\s+-\s+(\w+)"
        for o, count in Counter(
            re.findall(existential_pattern, self.pddl_goal)
        ).items():
            for i in range(1, int(count) + 1):
                name = f"{o}-{i}"
                self.scene_objects[name]["type"] = o

                # initialise beliefs about the object's location
                self.scene_objects[name]["beliefs"] = {}

                # if not a receptacle, then it's location is unknown
                if "isReceptacle" not in self.scene_objects[name]:
                    # all receptacles are possible locations for an object
                    self.scene_objects[name]["beliefs"][
                        "inReceptacle"
                    ] = self.scene_receptacles.copy()

                if "lamp" in name:
                    self.scene_objects[name]["isLight"] = True

        self.actions_taken = []

    @staticmethod
    def process_obs(observation: str) -> dict:
        """
        No LLM version of process_obs prefered for $ efficiency.
        """
        json_dict = {}
        # check if the receptacle is closed
        closed_receptacle = re.search(r"The (\w+ \d+) is closed", observation)
        if closed_receptacle:
            return json_dict
        # find the receptacle
        receptacle = re.search(
            r"(On|In) the (\w+ \d+)|You open the (\w+ \d+)", observation
        )
        if receptacle:
            # get the receptacle from the right group
            receptacle_key = (
                receptacle.group(2) if receptacle.group(2) else receptacle.group(3)
            )
            receptacle_key = receptacle_key.replace(" ", "-")
            json_dict[receptacle_key] = []
            # check if there's nothing in the receptacle
            no_items = re.search(r"you see nothing", observation)
            if no_items:
                return json_dict
            # find items in the receptacle
            items = re.findall(r"a (\w+ \d+)", observation)
            for item in items:
                json_dict[receptacle_key].append(item.replace(" ", "-"))
        return json_dict

    @staticmethod
    def find_receptacles_from_scene_observation(scene_observation: str) -> list[str]:
        """
        Given an Alfworld initial scene observation, return a list of receptacles.
        """
        receptacles = re.findall(r"a (\w+ \d+)", scene_observation)
        receptacles = [recep.replace(" ", "-") for recep in receptacles]
        return receptacles

    @staticmethod
    def get_receptacle_attributes(receptacle: str) -> dict:
        """
        Given a receptacle, return a dictionary of attributes.
        """
        attributes = {}

        # predicate special types
        if "sink" in receptacle:
            attributes["isSink"] = True
        elif "microwave" in receptacle:
            attributes["isMicrowave"] = True
        elif "fridge" in receptacle:
            attributes["isFridge"] = True

        # predicate openable
        if (
            "microwave" in receptacle
            or "fridge" in receptacle
            or "drawer" in receptacle
            or "cabinet" in receptacle
            or "safe" in receptacle
        ):
            attributes["openable"] = True

        attributes["type"] = receptacle.split("-")[0]
        attributes["isReceptacle"] = True

        return attributes

    @staticmethod
    def convert_pddl_action_to_alfworld(
        action_name: str, action_args: list[str]
    ) -> str:
        """
        Given a PDDL action, convert it to an Alfworld textworld action.
        """
        match action_name:
            case "examineobjectinlight":
                out = f"use {action_args[1]}"
            case "gotoreceptacle":
                out = f"go to {action_args[0]}"
            case "openreceptacle":
                out = f"open {action_args[0]}"
            case "closereceptacle":
                out = f"close {action_args[0]}"
            case "pickupobjectfromreceptacle":
                out = f"take {action_args[0]} from {action_args[1]}"
            case "putobject":
                out = f"put {action_args[0]} in/on {action_args[1]}"
            case "cleanobject":
                out = f"clean {action_args[0]} with {action_args[1]}"
            case "heatobject":
                out = f"heat {action_args[0]} with {action_args[1]}"
            case "coolobject":
                out = f"cool {action_args[0]} with {action_args[1]}"
            case _:
                raise ValueError(f"Unknown action: {action_name}")
        return out.replace("-", " ")

    def get_pddl_goal(self) -> str:
        """
        Given a task description return the PDDL goal using an LLM.
        """
        prompt_messages = GENERATE_GOAL_PROMPT + [
            {
                "role": "user",
                "content": self.task_description,
            }
        ]
        pddl_goal, token_usage = llm_cache(prompt_messages, stop=None)
        self.llm_tokens_sent += token_usage["total_tokens"]
        return pddl_goal

    def get_pddl_belief_predicate(
        self, init_str: str, belief_predicate: str, belief_values: list[str], top_n: 1
    ) -> list[str]:
        """
        Given a task description return the PDDL goal using an LLM.
        """
        user_prompt = (
            f"Predict: {belief_predicate}"
            + f"Select the top {top_n} likely items for ? from the list: {belief_values}"
            + "Return a parsable python list of choices."
        )
        prompt_messages = [
            {"role": "system", "content": f"Observed Environment\n{init_str}"},
            {"role": "user", "content": user_prompt},
        ]
        selected_values, token_usage = llm_cache(prompt_messages, stop=None)
        self.llm_tokens_sent += token_usage["total_tokens"]
        # parse the selected values as list
        try:
            selected_values = re.findall(r"'(.*?)'", selected_values)
        except Exception as e:
            self.logger.info(f"Error parsing selected values: {selected_values}")
            raise e
        return selected_values

    def get_pddl_objects(self) -> str:
        # get all objects/receptacles in scene
        objects_str = "".join(
            [f"{o} - {atts['type']}\n" for o, atts in self.scene_objects.items()]
        )
        return f"(:objects {objects_str})\n"

    def get_pddl_init(self) -> list[str]:
        # fill in known predicates from observation
        known_predicates = ""

        # known predicates
        for r, atts in self.scene_objects.items():
            for att, val in atts.items():
                if att in ["type", "beliefs"] or val is False:
                    continue
                if val is True:
                    known_predicates += f"({att} {r})\n"
                else:
                    known_predicates += f"({att} {r} {val})\n"

        # dynamic predicates (World Beliefs)
        belief_predicates = [known_predicates] * self.top_n
        for o, atts in self.scene_objects.items():
            if "beliefs" in atts:
                for belief_attribute in atts["beliefs"]:
                    options = atts["beliefs"][belief_attribute]

                    # sample N different worlds for each belief
                    if self.use_llm_search:
                        # Use LLM to guess which receptacle
                        sampled_beliefs = self.get_pddl_belief_predicate(
                            init_str=known_predicates,
                            belief_predicate=f"({belief_attribute} {o} ?)",
                            belief_values=options,
                            top_n=self.top_n,
                        )
                        # ensure that the sampled belief is in the list of options
                        hallucination_set = set(sampled_beliefs) - set(options)
                        for i, element in enumerate(sampled_beliefs):
                            if element in hallucination_set:
                                self.logger.warning(
                                    f"Hallucination: Sampled belief {element} not in {options}"
                                )
                                sampled_beliefs[i] = random.choice(options)
                    else:
                        sampled_beliefs = random.choices(options, k=self.top_n)

                    # append the sampled belief predicate to each world state
                    belief_predicates = list(
                        map(
                            lambda x, s: x + f"({belief_attribute} {o} {s})\n",
                            belief_predicates,
                            sampled_beliefs,
                        )
                    )

        return list(
            set([f"(:init {predicates})\n" for predicates in belief_predicates])
        )

    def get_pddl_problem(self) -> list[str]:
        # get n different init configurations
        inits = self.get_pddl_init()

        problems = []
        for init in inits:
            # construct to PDDL problem.pddl
            problems.append(
                "(define (problem alf)\n(:domain alfred)\n"
                + f"{self.get_pddl_objects()}{init}{self.pddl_goal})"
            )
        return problems

    def update_observation(self, observation: str) -> bool:
        # case for initial observation
        if observation == "":
            return True

        scene_obs = {}
        scene_changed = False

        # use last action to update scene_objects
        action_args = self.actions_taken[-1]
        action_name = action_args[0]
        action_args = action_args[1:]

        # we use the last action to update the scene
        # NOTE: this is using the symbolic :effects of the action
        #       as described in the PDDL domain
        match action_name:
            case "examineobjectinlight":
                self.scene_objects[action_args[0]]["examined"] = action_args[1]
            case "gotoreceptacle":
                for receptacle in self.scene_objects:
                    self.scene_objects[receptacle]["atReceptacleLocation"] = False
                self.scene_objects[action_args[0]]["atReceptacleLocation"] = True
                scene_obs = self.process_obs(observation)
                scene_changed = len(scene_obs) > 0
            case "openreceptacle":
                self.scene_objects[action_args[0]]["opened"] = True
                scene_obs = self.process_obs(observation)
                scene_changed = len(scene_obs) > 0
            case "closereceptacle":
                self.scene_objects[action_args[0]]["opened"] = False
            case "pickupobjectfromreceptacle":
                del self.scene_objects[action_args[0]]["inReceptacle"]
                self.scene_objects[action_args[0]]["holds"] = True
            case "putobject":
                self.scene_objects[action_args[0]]["holds"] = False
                self.scene_objects[action_args[0]]["inReceptacle"] = action_args[1]
            case "coolobject":
                self.scene_objects[action_args[0]]["isCool"] = True
                self.scene_objects[action_args[0]]["isHot"] = False
            case "heatobject":
                self.scene_objects[action_args[0]]["isHot"] = True
                self.scene_objects[action_args[0]]["isCool"] = False
            case "cleanobject":
                self.scene_objects[action_args[0]]["isClean"] = True

        # use observation to update scene_objects
        for receptacle, seen_objects in scene_obs.items():
            # if you can see objects in receptacle, it must be opened
            if "openable" in self.scene_objects[receptacle]:
                self.scene_objects[receptacle]["opened"] = True

            # update belief
            # all objects not observed at this receptacle cannot be believed to be in it
            for obj in self.scene_objects.keys():
                if (
                    obj not in seen_objects
                    and "beliefs" in self.scene_objects[obj]
                    and "inReceptacle" in self.scene_objects[obj]["beliefs"]
                    and receptacle in self.scene_objects[obj]["beliefs"]["inReceptacle"]
                ):
                    self.scene_objects[obj]["beliefs"]["inReceptacle"].remove(
                        receptacle
                    )

            # update inReceptacle for all objects observed at this receptacle
            for obj in seen_objects:
                self.scene_objects[obj]["type"] = obj.split("-")[0]
                self.scene_objects[obj]["inReceptacle"] = receptacle
                if (
                    "beliefs" in self.scene_objects[obj]
                    and "inReceptacle" in self.scene_objects[obj]["beliefs"]
                ):
                    del self.scene_objects[obj]["beliefs"]["inReceptacle"]
                if "lamp" in obj:
                    self.scene_objects[obj]["isLight"] = True

        return scene_changed

    def get_plan(self) -> list[str]:
        problems = self.get_pddl_problem()
        plans = parallel_lapkt_solver(problems, logger=self.logger)
        # greedy selection strategy
        return min(plans, key=len)

    def take_action(self, last_observation="") -> str:
        # sometimes move action doesn't trigger observation (e.g. if already close)
        if (
            last_observation == "Nothing happens."
            and self.actions_taken[-1][0] == "gotoreceptacle"
        ):
            return f"examine {self.actions_taken[-1][1]}".replace("-", " ")
        # if last action was not move, then we should have observed something
        elif last_observation == "Nothing happens.":
            raise Exception(f"Action {self.actions_taken[-1][0]} failed.")

        # update scene_objects with last observation
        changed = self.update_observation(last_observation)

        # if env changed, replan
        if changed:
            self.plan = self.get_plan()

        # get next action from plan
        pddl_action = self.plan.pop(0)
        # remove parentheses
        action_args = pddl_action.split(" ")

        # append action args to list of actions taken
        self.actions_taken.append(action_args)

        alfworld_action = self.convert_pddl_action_to_alfworld(
            action_args[0], action_args[1:]
        )

        return alfworld_action


if __name__ == "__main__":
    import alfworld
    import alfworld.agents.environment

    os.makedirs(LLMDPConfig.output_dir, exist_ok=True)
    run_name = f"{LLMDPConfig.output_dir}/{LLMDPConfig.name}_{LLMDPConfig.top_n}_{LLMDPConfig.llm_model}"

    with open(f"{LLMDPConfig.alfworld_config_path}/base_config.yaml") as reader:
        config = yaml.safe_load(reader)

    split = "eval_out_of_distribution"

    # UPDATE PATH TO ALFWORLD DATA
    for k in config:
        for i, j in config[k].items():
            if type(j) == str and j.startswith("$"):
                config[k][i] = config[k][i].replace(
                    "$ALFWORLD_DATA", LLMDPConfig.alfworld_data_path
                )

    env = getattr(alfworld.agents.environment, config["env"]["type"])(
        config, train_eval=split
    )
    env = env.init_env(batch_size=1)

    def process_ob(ob):
        if ob.startswith("You arrive at loc "):
            ob = ob[ob.find(". ") + 2 :]
        return ob

    NUM_GAMEFILES = len(env.gamefiles)

    logger = get_logger(f"{run_name}.log")

    def llmdp_run(agent) -> tuple[int, int]:
        last_observation = ""
        for i in range(1, 50):
            try:
                action = agent.take_action(last_observation=last_observation)
            except Exception as e:
                logger.error(f"Error in taking action {str(e)}")
                logger.info(str(agent.scene_objects))
                logger.info(str(agent.plan))
                return 0, i

            logger.info(f"{i} Action: {action}")
            observation, reward, done, info = env.step([action])
            observation, reward, done = (
                process_ob(observation[0]),
                info["won"][0],
                done[0],
            )
            last_observation = observation
            logger.info(f"{i} Obs: {last_observation}")
            if done:
                return reward, i
        return 0, i

    prefixes = {
        "pick_and_place": "put",
        "pick_clean_then_place": "clean",
        "pick_heat_then_place": "heat",
        "pick_cool_then_place": "cool",
        "look_at_obj": "examine",
        "pick_two_obj": "puttwo",
    }
    cnts = [0] * 6
    rs = [0] * 6
    results = []

    # load previous results (if llm errors - useful for resuming)
    prev_results = []
    # with open(f"{run_name}.json", "rb") as f:
    # prev_results = json.loads(f.read())

    for n in range(NUM_GAMEFILES):
        # Set seed for reproducibility
        random.seed(LLMDPConfig.seed)

        ob, info = env.reset()
        ob = "\n".join(ob[0].split("\n\n")[1:])
        scene_observation, task_description = ob.split("\n")
        name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])

        logger.info(name)
        logger.info(scene_observation)
        logger.info(task_description)

        # only run if prev failed (or if not in prev)
        if n < len(prev_results) and prev_results[n]["success"]:
            results.append(prev_results[n])
            continue

        for i, (k, v) in enumerate(prefixes.items()):
            if name.startswith(k):
                try:
                    agent = LLMDPAgent(
                        scene_observation,
                        task_description,
                        logger=logger,
                        use_llm_search=LLMDPConfig.use_llm_search,
                        top_n=LLMDPConfig.top_n,
                    )
                    r, length = llmdp_run(agent)
                except Exception as e:
                    logger.error(f"{str(e)}")
                    r = 0
                    length = 0

                rs[i] += r
                cnts[i] += 1

                results.append(
                    {
                        "task": v,
                        "success": r,
                        "length": length,
                        "llm_tokens": agent.llm_tokens_sent,
                        "llm_chat_tokens": agent.llm_chat_tokens_sent,
                    }
                )
                logger.info(f"# Tokens used: {agent.llm_tokens_sent}")
                out_log = f"# {n + 1} r: {r} rs: {rs} cnts: {cnts} sum(rs) / sum(cnts): {sum(rs) / sum(cnts)}"
                logger.info(out_log)
                break

        logger.info("------------\n")
        # save results
        with open(f"{run_name}.json", "w") as f:
            json.dump(results, f)
