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
    Minimal tree-guided counselor: the full topic graph is passed into each API call.
    Topic stack is updated from a small JSON navigation step, then one MI utterance is generated.
    """

    def __init__(self, goal, behavior, model, topic_graph=None, **kwargs):
        self.goal = goal
        self.behavior = behavior
        self.model = model
        manual_client = kwargs.get("manual_client", False)
        self.topic_graph = topic_graph if topic_graph is not None else DEFAULT_TOPIC_GRAPH
        self._validate_graph()

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
        user = (
            f"Full topic tree (stay coherent with these themes; do not invent nodes outside this JSON):\n{tree}\n\n"
            f"Current focus topic: {focus_topic}\n\n"
            f"Dialogue so far:\n{recent}\n\n"
            "Write the counselor's next line only, following motivational interviewing style."
        )
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
