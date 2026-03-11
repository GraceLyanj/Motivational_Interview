from typing import List, Literal, TypedDict, Optional

import json
import os
import random

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agents import Env, CAMI, Client
from agents.env import heuristic_moderator


class IncomingMessage(TypedDict):
  role: Literal["client", "counselor"]
  text: str


class AutoSessionRequest(TypedDict, total=False):
  profile_index: int
  max_turns: int


class AutoSessionResponse(TypedDict):
  conversation: List[IncomingMessage]


app = FastAPI(title="CAMI Chat Backend")

app.add_middleware(
  CORSMiddleware,
  # In development, allow all origins so the Vite dev server can call us
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


def run_cami_session(
  model: str = "gpt-5.2",
  retriever_path: str = "facebook/dpr-ctx_encoder-single-nq-base",
  wikipedia_dir: str = "./wikipedias",
  profile_path: str = "./annotations/profiles.jsonl",
  profile_index: Optional[int] = None,
  max_turns: int = 20,
) -> List[IncomingMessage]:
  """Run a single CAMI client–counselor session and return the conversation."""
  with open(profile_path, encoding="utf-8") as f:
    lines = f.readlines()

  if not lines:
    raise RuntimeError("No profiles found in profiles.jsonl")

  if profile_index is None:
    profile_index = random.randrange(len(lines))
  else:
    profile_index = max(0, min(profile_index, len(lines) - 1))

  sample = json.loads(lines[profile_index])

  goal = sample["topic"]
  behavior = sample["Behavior"]
  counselor = CAMI(goal=goal, behavior=behavior, model=model)

  reference = ""
  for speaker, utterance in zip(sample["speakers"][:50], sample["utterances"][:50]):
    if speaker == "client":
      reference += f"Client: {utterance}\n"
    else:
      reference += f"Counselor: {utterance}\n"

  client = Client(
    goal=sample["topic"],
    behavior=sample["Behavior"],
    reference=reference,
    personas=sample["Personas"],
    initial_stage=sample["states"][0],
    final_stage=sample["states"][-1],
    motivation=sample["Motivation"],
    beliefs=sample["Beliefs"],
    plans=sample["Acceptable Plans"],
    receptivity=sum(sample["suggestibilities"]) / len(sample["suggestibilities"]),
    model=model,
    wikipedia_dir=wikipedia_dir,
    retriever_path=retriever_path,
  )

  env = Env(
    client=client,
    counselor=counselor,
    max_turns=max_turns,
    output_file=None,
  )

  # Run full interaction (both sides are AI)
  env.interact()

  conversation_messages: List[IncomingMessage] = []
  for line in env.conversation:
    text = line.strip()
    if text.startswith("Counselor:"):
      role: Literal["client", "counselor"] = "counselor"
      content = text[len("Counselor:") :].strip()
    elif text.startswith("Client:"):
      role = "client"
      content = text[len("Client:") :].strip()
    else:
      # Fallback: assign to counselor
      role = "counselor"
      content = text
    conversation_messages.append({"role": role, "text": content})

  return conversation_messages


def cami_stream_generator(
  model: str = "gpt-5.2",
  retriever_path: str = "facebook/dpr-ctx_encoder-single-nq-base",
  wikipedia_dir: str = "./wikipedias",
  profile_path: str = "./annotations/profiles.jsonl",
  profile_index: Optional[int] = None,
  max_turns: int = 20,
):
  """Yield CAMI dialogue messages one-by-one (for live updates)."""
  with open(profile_path, encoding="utf-8") as f:
    lines = f.readlines()

  if not lines:
    return

  if profile_index is None:
    profile_index = random.randrange(len(lines))
  else:
    profile_index = max(0, min(profile_index, len(lines) - 1))

  sample = json.loads(lines[profile_index])

  goal = sample["topic"]
  behavior = sample["Behavior"]
  counselor = CAMI(goal=goal, behavior=behavior, model=model)

  reference = ""
  for speaker, utterance in zip(sample["speakers"][:50], sample["utterances"][:50]):
    if speaker == "client":
      reference += f"Client: {utterance}\n"
    else:
      reference += f"Counselor: {utterance}\n"

  client = Client(
    goal=sample["topic"],
    behavior=sample["Behavior"],
    reference=reference,
    personas=sample["Personas"],
    initial_stage=sample["states"][0],
    final_stage=sample["states"][-1],
    motivation=sample["Motivation"],
    beliefs=sample["Beliefs"],
    plans=sample["Acceptable Plans"],
    receptivity=sum(sample["suggestibilities"]) / len(sample["suggestibilities"]),
    model=model,
    wikipedia_dir=wikipedia_dir,
    retriever_path=retriever_path,
  )

  conversation: List[str] = [
    "Counselor: Hello. How are you?",
    "Client: I am good. What about you?",
  ]

  def split_thinking_and_utterance(line: str):
    """Split a raw CAMI line into (thinking, role, utterance).

    """
    text = line.strip()
    role: Literal["client", "counselor"]

    thinking = None
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
      thinking_block = text[first_bracket : last_bracket + 1]
      thinking = thinking_block.strip()
      text = (text[:first_bracket] + text[last_bracket + 1 :]).strip()

    idx_client = text.find("Client:")
    idx_counselor = text.find("Counselor:")

    if idx_client == -1 and idx_counselor == -1:
      return thinking, "counselor", text

    if idx_client != -1 and (idx_counselor == -1 or idx_client < idx_counselor):
      role = "client"
      prefix = "Client:"
      start_idx = idx_client
    else:
      role = "counselor"
      prefix = "Counselor:"
      start_idx = idx_counselor

    pre_speaker = text[:start_idx].strip()
    if pre_speaker:
      if thinking:
        thinking = f"{pre_speaker} {thinking}"
      else:
        thinking = pre_speaker

    utterance = text[start_idx + len(prefix) :].strip()

    return thinking, role, utterance

  # Yield initial context (no thinking section)
  for line in conversation:
    _, role, utterance = split_thinking_and_utterance(line)
    yield {
      "kind": "message",
      "role": role,
      "text": utterance,
    }

  for _ in range(max_turns):
    # Counselor turn
    counselor_response = counselor.reply().replace("\n", " ")
    conversation.append(counselor_response)
    client.receive(counselor_response)

    thinking, role, utterance = split_thinking_and_utterance(counselor_response)
    if thinking:
      yield {
        "kind": "thinking",
        "role": role,
        "text": thinking,
      }
    yield {
      "kind": "message",
      "role": role,
      "text": utterance,
    }

    if heuristic_moderator(conversation):
      break

    # Client turn
    client_response = client.reply().replace("\n", " ")

    thinking_c, role_c, utterance_c = split_thinking_and_utterance(client_response)
    if thinking_c:
      yield {
        "kind": "thinking",
        "role": role_c,
        "text": thinking_c,
      }
    yield {
      "kind": "message",
      "role": role_c,
      "text": utterance_c,
    }

    if (
      "You are motivated because" in client_response
      or "You should highlight current state and engagement, express a desire to end the current session"
      in client_response
    ):
      break

    conversation.append(client_response)
    counselor.receive(client_response)

    if heuristic_moderator(conversation):
      break


@app.post("/auto_session")
def auto_session(request: AutoSessionRequest | None = None) -> AutoSessionResponse:
  """Run a CAMI auto session (AI client + AI counselor) and return the dialogue."""
  profile_index = None
  max_turns = 20
  if request is not None:
    profile_index = request.get("profile_index")
    max_turns = request.get("max_turns", max_turns)

  conversation = run_cami_session(
    model=os.environ.get("OPENAI_MODEL", "gpt-5.2"),
    retriever_path="facebook/dpr-ctx_encoder-single-nq-base",
    wikipedia_dir="./wikipedias",
    profile_path="./annotations/profiles.jsonl",
    profile_index=profile_index,
    max_turns=max_turns,
  )

  return {"conversation": conversation}


@app.get("/auto_session_stream")
def auto_session_stream() -> StreamingResponse:
  """Stream a CAMI auto session as server-sent events for live UI updates."""

  def event_stream():
    for msg in cami_stream_generator(
      model=os.environ.get("OPENAI_MODEL", "gpt-5.2"),
      retriever_path="facebook/dpr-ctx_encoder-single-nq-base",
      wikipedia_dir="./wikipedias",
      profile_path="./annotations/profiles.jsonl",
      profile_index=None,
      max_turns=20,
    ):
      yield f"data: {json.dumps(msg)}\n\n"

  return StreamingResponse(event_stream(), media_type="text/event-stream")

