import json
import os
import random

from openai import OpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from .topic_graph_default import DEFAULT_TOPIC_GRAPH, ROOT_TOPICS

PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
if not PORTKEY_API_KEY:
    raise RuntimeError(
        "Missing PORTKEY_API_KEY. Set it in your shell or in .env before starting backend."
    )

_client = OpenAI(
    api_key="portkey",
    base_url=PORTKEY_GATEWAY_URL,
    default_headers=createHeaders(
        api_key=PORTKEY_API_KEY,
        provider="@28935-openai",
    ),
)


def _chat(messages, model, temperature=0.6, top_p=0.9, max_completion_tokens=200):
    return _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
    ).choices[0].message.content


def _chat_json(messages, model):
    return _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        top_p=0.1,
        max_completion_tokens=300,
        response_format={"type": "json_object"},
    ).choices[0].message.content


def _tree_for_prompt(topic_graph: dict) -> str:
    return json.dumps(topic_graph, indent=2, ensure_ascii=False)


class CAMI:
    """
    Tree-guided counselor: the full topic graph is passed into each API call.
    Topic stack is updated from a small JSON navigation step, then one MI utterance is generated.
    """

    def __init__(self, goal, behavior, model, topic_graph=None, **kwargs):
        self.goal = goal
        self.behavior = behavior
        self.model = model
        manual_client = kwargs.get("manual_client", False)
        self.topic_graph = topic_graph if topic_graph is not None else DEFAULT_TOPIC_GRAPH
        self._validate_graph()

        # Topic descriptions (used to ground the focused topic each turn).
        self.topic2description = {
            # --- ROOT LEVEL ---
            "Autonomy": (
                f"You can explore how {behavior} currently feels forced or dictated by others[cite: 78, 79]. "
                f"You can also discuss how {goal} establishes long-term intrinsic motivation by allowing actions to align with personal values, interests, and independent decisions[cite: 79, 80]."
            ),
            # --- PRIMARY SUB-NODES ---
            "Sense of Choice": (
                f"You can explore how {behavior} lacks a feeling of true decision-making power[cite: 81, 82]. "
                f"You can also emphasize how {goal} provides a genuine sense of choice, which reduces psychological resistance and boosts overall engagement."
            ),
            "Perceived Locus of Causality": (
                f"You can explore whether {behavior} is perceived as being driven by internal (self-chosen) or external (controlled) factors[cite: 30, 75, 76]. "
                f"You can also highlight how {goal} identifies the current motivational stage to guide the user toward a more self-determined drive for activity[cite: 77]."
            ),
            "Control of Exercise Plan": (
                f"You can explore how {behavior} is hindered by rigid logistics[cite: 31, 54]. "
                f"You can also discuss how {goal} allows the user to dictate workout logistics, such as time, place, and type of exercise, to remove practical barriers to execution[cite: 37, 38, 39, 55, 56]."
            ),
            # --- PSYCHOLOGICAL & IDENTITY ALIGNMENT ---
            "Consistent with Personal Identity": (
                f"You can explore how {behavior} feels like an external chore that doesn't fit the user[cite: 34, 66]. "
                f"You can also highlight how {goal} makes exercise a natural reflection of 'who I am,' transforming movement into a default lifestyle habit[cite: 67, 68]."
            ),
            "Alignment with Personal Value": (
                f"You can explore how {behavior} might conflict with core virtues[cite: 36, 69]. "
                f"You can also discuss how {goal} transforms fitness into a pathway for realizing self-worth and demonstrating values like self-discipline and resilience[cite: 70, 71]."
            ),
            # --- EXTERNAL PRESSURE & REGULATION ---
            "No External Pressure": (
                f"You can explore how {behavior} is driven by coercion from peers or societal expectations[cite: 35, 57, 58]. "
                f"You can also discuss how {goal} shifts the mindset from 'I have to' to 'I want to' by removing external pressure[cite: 59]."
            ),
            "Absence of Conditional Punishment": (
                f"You can explore how {behavior} is motivated by fear, guilt, or shame[cite: 32, 60, 61]. "
                f"You can also highlight how {goal} prevents negative emotional associations with physical activity by ensuring exercise is not a response to self-punishment[cite: 62]."
            ),
            "Free of External Awards": (
                f"You can explore how {behavior} depends solely on praise or tangible rewards[cite: 33, 63, 64]. "
                f"You can also discuss how {goal} cultivates joy and satisfaction in the activity itself rather than chasing external validation[cite: 65]."
            ),
            "Driven by Health Outcomes": (
                f"You can explore how {behavior} might be neglecting long-term wellness[cite: 40, 72]. "
                f"You can also emphasize how {goal} leverages a personal desire to improve physical energy and functionality as a powerful internal motivator[cite: 73, 74]."
            ),
            # --- LOGISTICAL SUB-NODES UNDER "CONTROL OF EXERCISE PLAN" ---
            "Time": (
                f"You can explore how rigid scheduling in {behavior} creates friction in your client's daily routine[cite: 37, 51, 54]. "
                f"You can also discuss how achieving {goal} through flexible timing allows them to integrate activity whenever it best fits their energy and lifestyle[cite: 37, 51, 54]."
            ),
            "Place": (
                f"You can explore how the current environment in {behavior} might feel inconvenient or uncomfortable[cite: 38, 52, 55]. "
                f"You can also highlight how {goal} gives them the power to choose a location—whether at home, outdoors, or a gym—that feels most conducive to their success[cite: 38, 52, 55]."
            ),
            "Type of Exercise": (
                f"You can explore how {behavior} might involve activities that the client finds boring or unsuitable[cite: 39, 53, 56]. "
                f"You can also discuss how {goal} empowers them to select the specific forms of movement they actually enjoy, ensuring the workout feels like a choice rather than a chore[cite: 39, 53, 56]."
            ),
        }

        self.system_prompt = (
            f"You are a motivational interviewing counselor. Counseling goal: {goal}. "
            f"Target behavior: {behavior}. Be warm and brief (under 50 words per turn). "
            "Do not mention 'motivational interviewing' by name. Start counselor lines with 'Counselor: '."
        )
        first_counselor = "Counselor: Hello. How are you?"
        first_client = "Client: I am good. What about you?"
        if manual_client:
            self.conversation = [first_counselor]
            self.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "assistant", "content": first_counselor},
            ]
        else:
            self.conversation = [first_counselor, first_client]
            self.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "assistant", "content": first_counselor},
                {"role": "user", "content": first_client},
            ]
        self.topic_stack: list[str] = []
        self.explored_topics: list[str] = []

    def _validate_graph(self):
        for name, meta in self.topic_graph.items():
            if "Children" not in meta or "Parent" not in meta:
                raise ValueError(f'Topic graph node "{name}" must have Parent and Children keys')

    def _nav_choices(self):
        """Return allowed (step_into, switch, step_out) topic lists; None if that move is invalid."""
        g = self.topic_graph
        stack = self.topic_stack
        if not stack:
            return ROOT_TOPICS.copy(), None, None
        if len(stack) == 1:
            return g[stack[0]]["Children"].copy(), ROOT_TOPICS.copy(), None
        if len(stack) == 2:
            return (
                g[stack[1]]["Children"].copy(),
                g[stack[0]]["Children"].copy(),
                ROOT_TOPICS.copy(),
            )
        switch = g[stack[1]]["Children"].copy()
        step_out = g[stack[0]]["Children"].copy()
        return None, switch, step_out

    def _current_focus_topic(self) -> str:
        if self.topic_stack:
            return self.topic_stack[-1]
        return "undecided_root"

    def _apply_navigation(self, action: str, topic: str | None) -> str:
        """Update topic_stack. Returns normalized action name for logging."""
        a = (action or "stay").strip().lower().replace(" ", "_")
        step_in, switch, step_out = self._nav_choices()

        def pick(valid: list[str] | None) -> str | None:
            if not valid or not topic or topic not in valid:
                return None
            return topic

        if not self.topic_stack:
            if a in ("choose_root", "step_into", "switch", "initialize"):
                t = pick(ROOT_TOPICS) or random.choice(ROOT_TOPICS)
                self.topic_stack = [t]
                return "initialize"
            self.topic_stack = [random.choice(ROOT_TOPICS)]
            return "initialize"

        if a == "stay" or a == "remain":
            return "stay"

        if a in ("step_into", "stepinto", "deeper") and step_in:
            t = pick(step_in) or (random.choice(step_in) if step_in else None)
            if t:
                self.topic_stack.append(t)
                return "step_into"

        if a in ("switch",) and switch:
            t = pick(switch) or random.choice(switch)
            self.topic_stack.pop()
            self.topic_stack.append(t)
            return "switch"

        if a in ("step_out", "stepout", "back_up", "back") and step_out:
            t = pick(step_out) or random.choice(step_out)
            self.topic_stack.pop()
            self.topic_stack.pop()
            self.topic_stack.append(t)
            return "step_out"

        return "stay"

    def _parse_nav_json(self, raw: str) -> tuple[str, str | None]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return "stay", None
        action = str(data.get("action", "stay"))
        topic = data.get("topic")
        if topic is not None:
            topic = str(topic)
        return action, topic

    def _navigate_via_api(self) -> tuple[str, str]:
        tree = _tree_for_prompt(self.topic_graph)
        path = " -> ".join(self.topic_stack) if self.topic_stack else "(no topic yet — pick one root)"
        step_in, switch, step_out = self._nav_choices()
        recent = "\n".join(self.conversation[-8:])

        lines = [
            "Topic tree (JSON). Each node has Parent and Children; only use topic names that appear here.",
            tree,
            "",
            f"Current path: {path}",
            f"Counseling goal: {self.goal}; behavior: {self.behavior}",
            "",
            "Recent dialogue:",
            recent,
            "",
            "Choose how to adjust the topic focus for the counselor's NEXT reply.",
        ]
        if not self.topic_stack:
            lines.append(f'- "initialize": set path to one of {ROOT_TOPICS}')
        else:
            if step_in:
                lines.append(
                    f'- "step_into": go to a child of the current leaf — topic must be one of {step_in}'
                )
            if switch:
                lines.append(
                    f'- "switch": replace the current branch under the same parent — topic must be one of {switch}'
                )
            if step_out:
                lines.append(
                    f'- "step_out": go up one level then choose a sibling under the root — topic must be one of {step_out}'
                )
            lines.append('- "stay": keep the current leaf topic (topic must equal the current leaf name)')

        lines.append(
            'Reply with JSON only: {"action":"<initialize|step_into|switch|step_out|stay>","topic":"<exact name from tree or null>"}'
        )

        raw = _chat_json(
            [{"role": "user", "content": "\n".join(lines)}],
            self.model,
        )
        action, topic = self._parse_nav_json(raw)
        applied = self._apply_navigation(action, topic)
        focus = self._current_focus_topic()
        return applied, focus

    def _generate_utterance(self, focus_topic: str) -> str:
        tree = _tree_for_prompt(self.topic_graph)
        recent = "\n".join(self.conversation[-10:])
        focus_desc = self.topic2description.get(focus_topic, "")
        parts: list[str] = [
            "Full topic tree (stay coherent with these themes; do not invent nodes outside this JSON):\n",
            tree,
            "\n\n",
            f"Current focus topic: {focus_topic}\n\n",
        ]
        if focus_desc:
            parts.extend(["Focus topic description:\n", focus_desc, "\n\n"])
        parts.extend(
            [
                "Dialogue so far:\n",
                recent,
                "\n\n",
                "Write the counselor's next line only, following motivational interviewing style.",
            ]
        )
        user = "".join(parts)
        text = _chat(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ],
            self.model,
            temperature=0.7,
            max_completion_tokens=200,
        )
        text = " ".join((text or "").splitlines()).strip()
        text = text.replace("*", "").replace("#", "")
        if not text.startswith("Counselor:"):
            text = f"Counselor: {text.lstrip()}"
        if "Client:" in text:
            text = text.split("Client:")[0].strip()
        return text

    def receive(self, response: str):
        self.messages.append({"role": "user", "content": response})
        self.conversation.append(response)

    def reply(self):
        nav_action, focus = self._navigate_via_api()
        self.explored_topics.append(focus)
        utterance = self._generate_utterance(focus)
        self.messages.append({"role": "assistant", "content": utterance})
        self.conversation.append(utterance)
        return f"[Nav: {nav_action} || Topic: {focus}] {utterance}"
