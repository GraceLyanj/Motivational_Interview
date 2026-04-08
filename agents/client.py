import backoff
import openai
from openai import OpenAI
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DPRContextEncoder
import heapq
import random
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from .topic_graph_default import DEFAULT_TOPIC_GRAPH

PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
if not PORTKEY_API_KEY:
    raise RuntimeError(
        "Missing PORTKEY_API_KEY. Set it in your shell or in .env before starting backend."
    )

client = OpenAI(
    api_key="portkey",   # placeholder
    base_url=PORTKEY_GATEWAY_URL,
    default_headers=createHeaders(
        api_key=PORTKEY_API_KEY,
        provider="@28935-openai"
    )
)

def get_precise_response(messages, model="gpt-5.2", temperature=0.2, top_p=0.1):
    message = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
    )
    return message.choices[0].message.content


def get_chatbot_response(
    messages, model="gpt-5.2", temperature=0.7, top_p=0.8, max_tokens=100
):

    
    message = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=100,
        
    )
    return message.choices[0].message.content

def get_json_response(messages, model="gpt-5.2", temperature=0.2, top_p=0.1):
    if "json" not in messages[0]["content"].lower():
        messages[0]["content"] += " Please respond in JSON format."
    message = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        response_format={"type": "json_object"},
    )
    return message.choices[0].message.content


class Client:
    def __init__(
        self,
        goal,
        behavior,
        reference,
        personas,
        initial_stage,
        final_stage,
        motivation,
        beliefs,
        plans,
        receptivity,
        model,
        wikipedia_dir,
        retriever_path
    ):
        self.goal = goal
        self.behavior = behavior
        self.personas = personas
        self.motivation = motivation[-1]
        self.engagemented_topics = motivation[:-1]
        self.beliefs = beliefs
        self.initial_stage = initial_stage
        self.state = initial_stage
        self.final_stage = final_stage
        self.acceptable_plans = plans
        self.receptivity = receptivity
        self.engagement = receptivity
        self.context = [
            "Counselor: Hello. How are you?",
            "Client: I am good. What about you?",
        ]
        self.action2prompt = {
            "Deny": "Deny the behavior is a problem.",
            "Downplay": "Minimize how much the behavior matters.",
            "Blame": "Blame circumstances or others.",
            "Inform": "Share a bit of personal context or feeling.",
            "Engage": "Stay in character; match the reference tone.",
            "Hesitate": "Sound unsure about changing.",
            "Doubt": "Doubt the plan will work; stay vague.",
            "Acknowledge": "Admit change may be needed.",
            "Accept": "Accept the suggested plan.",
            "Reject": "Reject the plan as a bad fit.",
            "Plan": "Offer concrete steps.",
            "Terminate": "Politely want to pause and continue another time.",
        }

        self.state2prompt = {
            "Precontemplation": f"You see {self.behavior} as fine for now.",
            "Contemplation": f"You see {self.behavior} as a problem but are torn about {self.goal}.",
            "Preparation": f"You are leaning into action toward {self.goal}.",
        }

        self.topic2description = {
            # --- ROOT LEVEL ---
            "Autonomy": (
                f"You can explore how {self.behavior} currently feels forced or dictated by others[cite: 78, 79]. "
                f"You can also discuss how {self.goal} establishes long-term intrinsic motivation by allowing actions to align with personal values, interests, and independent decisions[cite: 79, 80]."
            ),
            # --- PRIMARY SUB-NODES ---
            "Sense of Choice": (
                f"You can explore how {self.behavior} lacks a feeling of true decision-making power[cite: 81, 82]. "
                f"You can also emphasize how {self.goal} provides a genuine sense of choice, which reduces psychological resistance and boosts overall engagement."
            ),
            "Perceived Locus of Causality": (
                f"You can explore whether {self.behavior} is perceived as being driven by internal (self-chosen) or external (controlled) factors[cite: 30, 75, 76]. "
                f"You can also highlight how {self.goal} identifies the current motivational stage to guide the user toward a more self-determined drive for activity[cite: 77]."
            ),
            "Control of Exercise Plan": (
                f"You can explore how {self.behavior} is hindered by rigid logistics[cite: 31, 54]. "
                f"You can also discuss how {self.goal} allows the user to dictate workout logistics, such as time, place, and type of exercise, to remove practical barriers to execution[cite: 37, 38, 39, 55, 56]."
            ),
            # --- PSYCHOLOGICAL & IDENTITY ALIGNMENT ---
            "Consistent with Personal Identity": (
                f"You can explore how {self.behavior} feels like an external chore that doesn't fit the user[cite: 34, 66]. "
                f"You can also highlight how {self.goal} makes exercise a natural reflection of 'who I am,' transforming movement into a default lifestyle habit[cite: 67, 68]."
            ),
            "Alignment with Personal Value": (
                f"You can explore how {self.behavior} might conflict with core virtues[cite: 36, 69]. "
                f"You can also discuss how {self.goal} transforms fitness into a pathway for realizing self-worth and demonstrating values like self-discipline and resilience[cite: 70, 71]."
            ),
            # --- EXTERNAL PRESSURE & REGULATION ---
            "No External Pressure": (
                f"You can explore how {self.behavior} is driven by coercion from peers or societal expectations[cite: 35, 57, 58]. "
                f"You can also discuss how {self.goal} shifts the mindset from 'I have to' to 'I want to' by removing external pressure[cite: 59]."
            ),
            "Absence of Conditional Punishment": (
                f"You can explore how {self.behavior} is motivated by fear, guilt, or shame[cite: 32, 60, 61]. "
                f"You can also highlight how {self.goal} prevents negative emotional associations with physical activity by ensuring exercise is not a response to self-punishment[cite: 62]."
            ),
            "Free of External Awards": (
                f"You can explore how {self.behavior} depends solely on praise or tangible rewards[cite: 33, 63, 64]. "
                f"You can also discuss how {self.goal} cultivates joy and satisfaction in the activity itself rather than chasing external validation[cite: 65]."
            ),
            "Driven by Health Outcomes": (
                f"You can explore how {self.behavior} might be neglecting long-term wellness[cite: 40, 72]. "
                f"You can also emphasize how {self.goal} leverages a personal desire to improve physical energy and functionality as a powerful internal motivator[cite: 73, 74]."
            ),
            # --- LOGISTICAL SUB-NODES UNDER "CONTROL OF EXERCISE PLAN" ---
            "Time": (
                f"You can explore how rigid scheduling in {self.behavior} creates friction in your client's daily routine. "
                f"You can also discuss how achieving {self.goal} through flexible timing allows them to integrate activity whenever it best fits their energy and lifestyle[cite: 37, 51, 54]."
            ),
            "Place": (
                f"You can explore how the current environment in {self.behavior} might feel inconvenient or uncomfortable. "
                f"You can also highlight how {self.goal} gives them the power to choose a location—whether at home, outdoors, or a gym—that feels most conducive to their success[cite: 38, 52, 55]."
            ),
            "Type of Exercise": (
                f"You can explore how {self.behavior} might involve activities that the client finds boring or unsuitable. "
                f"You can also discuss how {self.goal} empowers them to select the specific forms of movement they actually enjoy, ensuring the workout feels like a choice rather than a chore[cite: 39, 53, 56]."
            ),
        }

        _w = 1
        self.topic_graph = {}
        _autonomy_children = DEFAULT_TOPIC_GRAPH["Autonomy"]["Children"]
        for child in _autonomy_children:
            self.topic_graph.setdefault("Autonomy", {})[child] = _w
            self.topic_graph.setdefault(child, {})["Autonomy"] = _w
        for logistical in ("Time", "Place", "Type of Exercise"):
            self.topic_graph.setdefault("Control of Exercise Plan", {})[logistical] = _w
            self.topic_graph.setdefault(logistical, {})["Control of Exercise Plan"] = _w

        self.all_topics = []
        for nodes in self.topic_graph:
            if nodes not in self.all_topics:
                self.all_topics.append(nodes)
            for node in self.topic_graph[nodes]:
                if node not in self.all_topics:
                    self.all_topics.append(node)
        self.passages = []
        for topic in self.all_topics:
            path = os.path.join(wikipedia_dir, topic)
            body = ""
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    body = f.read()
            self.passages.append(self.topic2description[topic] + body)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(
            retriever_path
        )
        self.retriever = DPRContextEncoder.from_pretrained(
            retriever_path
        ).to(device)
        self.retriever.eval()

        system_prompt = f"""You are the Client. Behavior: {self.behavior}. Counselor's goal: {self.goal}.

Persona notes:
[@personas]

Style reference (parallel dialogue—match length and voice, not necessarily facts):
{reference}

Rules: Start every reply with "Client: ". Follow only what square brackets in the user message ask. Stay consistent with persona. ≤3 short sentences; natural, not formal."""
        personas = "- " + "\n- ".join(self.personas) + "\n" + "\n".join(f"- {b}" for b in self.beliefs)
        system_prompt = system_prompt.replace("[@personas]", personas)
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Counselor: Hello. How are you?"},
            {"role": "assistant", "content": "Client: I am good. What about you?"},
        ]
        self.error_topic_count = 0
        self.model = model

    def verify_motivation(self):
        prompt = """Does the LAST counselor line speak to this client's specific motivation (topic, risk/benefit, who it affects)?

Goal: [@goal]
Motivation: [@motivation]

Recent lines:
- [@context]

Reply with exactly one line: Answer: Yes
or Answer: No
Optionally one short reason on a second line.
"""     
        prompt = prompt.replace("[@goal]", self.goal)
        prompt = prompt.replace("[@context]", "\n- ".join(self.context[-5:]))
        prompt = prompt.replace("[@motivation]", self.motivation)
        response = get_precise_response(
            messages=[{"role": "user", "content": prompt}], model=self.model
        )
        if "yes" in response.lower():
            self.state = "Motivation"
        return response.split("\n")[0].split(": ")[-1]

    def top5_related_topics(self):
        query = self.context[-1].split("Counselor: ")[-1]
        
        # 1. 取得 Query 的向量
        with torch.no_grad():
            query_inputs = self.retriever_tokenizer(
                query, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=512
            ).to(self.retriever.device)
            # DPR 輸出 pooler_output 作為向量表示
            query_emb = self.retriever(**query_inputs).pooler_output # (1, 768)

            # 2. 取得所有 Topics/Passages 的向量
            # 注意：這裡假設 self.passages 是已經處理好的文字清單
            passage_inputs = self.retriever_tokenizer(
                self.passages, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=512
            ).to(self.retriever.device)
            passage_embs = self.retriever(**passage_inputs).pooler_output # (N, 768)

            # 3. 計算相似度分數 (內積)
            # 矩陣相乘：(1, 768) x (768, N) = (1, N)
            batch_scores = torch.matmul(query_emb, passage_embs.transpose(0, 1)).view(-1)
            
            # 使用 sigmoid 轉換為 0~1 的機率（非必要但符合原代碼邏輯）
            scores_sigmoid = torch.sigmoid(batch_scores)
            scores = scores_sigmoid.tolist()

        # 4. 取得前 5 名
        top_5_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:5]
        top5_topics = [self.all_topics[idx] for idx in top_5_indices]
        return top5_topics

    def dijkstra(self, graph, start_node, target_node):
        # Initialize distances dictionary with infinity for all nodes
        distances = {node: float("infinity") for node in graph}
        distances[start_node] = 0

        # Priority queue to store (distance, node)
        pq = [(0, start_node)]

        # Keep track of visited nodes
        visited = set()

        while pq:
            # Get node with minimum distance
            current_distance, current_node = heapq.heappop(pq)

            # If we reached target node, return the distance
            if current_node == target_node:
                return current_distance

            # Skip if we've already visited this node
            if current_node in visited:
                continue

            visited.add(current_node)

            # Check all neighbors
            for neighbor, weight in graph[current_node].items():
                if neighbor not in visited:
                    distance = current_distance + weight

                    # If we found a shorter path, update it
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(pq, (distance, neighbor))

        # If no path found
        return float("infinity")

    def update_state(self):
        if self.state == "Contemplation":
            if len(self.beliefs) == 0:
                self.state = "Preparation"
            return
        elif self.state == "Preparation":
            return
        else:
            top_topics = self.top5_related_topics()
            predicted_topic = top_topics[0]
            anchor = self.engagemented_topics[0]
            if anchor not in self.topic_graph or predicted_topic not in self.topic_graph:
                self.engagement = 1
                if len(self.context) > 10:
                    self.error_topic_count += 1
                return f"The client's perceived topic is {predicted_topic}."
            if predicted_topic == anchor:
                self.engagement = 4
                self.error_topic_count = 0
                motivation_analysis = self.verify_motivation()
                return motivation_analysis
            distance = self.dijkstra(
                self.topic_graph, anchor, predicted_topic
            )
            if distance <= 3:
                self.engagement = 3
                self.error_topic_count = 0
                return f"The client's perceived topic is {predicted_topic}."
            if distance <= 5:
                self.engagement = 2
                return f"The client's perceived topic is {predicted_topic}."
            else:
                self.engagement = 1
                if len(self.context) > 10:
                    self.error_topic_count += 1
                return f"The client's perceived topic is {predicted_topic}."

    def select_action(self):
        prompt = """You are the client. Given the dialogue, assign probabilities (sum 100) for the client's next stance:
Deny | Downplay | Blame | Inform | Engage
(Deny=refuse it's a problem; Downplay=minimize; Blame=externalize; Inform=share feeling/fact; Engage=light rapport)

[@context]

JSON only, e.g. {"Deny":20,"Downplay":20,"Blame":20,"Inform":25,"Engage":15}
"""
        prompt = prompt.replace(
            "[@context]",
            "\n".join(self.context[-3:])
            .replace("Client:", "**Client**:")
            .replace("Counselor:", "**Counselor**:"),
        )
        context_aware_action_distribution = None
        for _ in range(5):
            response = get_json_response(
                messages=[{"role": "user", "content": prompt}], model=self.model
            )
            response = response.replace("```", "").replace("json", "")
            try:
                context_aware_action_distribution = eval(response)
            except SyntaxError:
                continue
            if context_aware_action_distribution:
                break
        if not context_aware_action_distribution:
            context_aware_action_distribution = {
                "Deny": 20,
                "Downplay": 20,
                "Blame": 20,
                "Engage": 20,
                "Inform": 20,
            }
        if self.receptivity < 2:
            receptivity_aware_action_distribution = {
                "Deny": 23,
                "Downplay": 28,
                "Blame": 15,
                "Engage": 11,
                "Inform": 22,
            }
        elif self.receptivity < 3:
            receptivity_aware_action_distribution = {
                "Deny": 20,
                "Downplay": 25,
                "Blame": 10,
                "Engage": 15,
                "Inform": 30,
            }
        elif self.receptivity < 4:
            receptivity_aware_action_distribution = {
                "Deny": 19,
                "Downplay": 21,
                "Blame": 11,
                "Engage": 13,
                "Inform": 36,
            }
        elif self.receptivity < 5:
            receptivity_aware_action_distribution = {
                "Deny": 9,
                "Downplay": 20,
                "Blame": 13,
                "Engage": 14,
                "Inform": 44,
            }
        else:
            receptivity_aware_action_distribution = {
                "Deny": 7,
                "Downplay": 13,
                "Blame": 4,
                "Engage": 16,
                "Inform": 60,
            }
        action_distribution = {
            action: context_aware_action_distribution.get(action, 0)
            + receptivity_aware_action_distribution[action]
            for action in receptivity_aware_action_distribution
        }
        if len(self.personas) == 0:
            action_distribution["Inform"] = 0
        if len(self.beliefs) == 0:
            action_distribution["Blame"] = 0
        # normalize
        action_distribution = {
            k: v / sum(action_distribution.values())
            for k, v in action_distribution.items()
        }
        sampled_action = np.random.choice(
            list(action_distribution.keys()),
            size=1,
            p=list(action_distribution.values()),
        )[0]
        return sampled_action

    def select_information(self, action):
        messages = []
        if "?" not in self.context[-1]:
            return None
        prompt = """Last counselor line a question? Yes or No.

[@conv]"""
        prompt = prompt.replace("[@conv]", "\n".join(self.context[-3:]))
        response = "Yes."
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": response})
        if action == "Inform":
            prompt2 = """Can this persona answer it? Yes or No
[@persona]"""
            personas = self.personas
        elif action == "Downplay":
            prompt2 = """Can this belief support a downplaying reply? Yes or No
[@persona]"""
            personas = self.beliefs
        elif action == "Blame":
            prompt2 = """Can this belief support blaming externals? Yes or No
[@persona]"""
            personas = self.beliefs
        elif action == "Hesitate":
            prompt2 = """Can this belief support a hesitant reply? Yes or No
[@persona]"""
            personas = self.beliefs
        for persona in personas:
            prompt = prompt2.replace("[@persona]", persona)
            messages.append({"role": "user", "content": prompt})
            response = get_precise_response(messages=messages, model=self.model)
            messages.append({"role": "assistant", "content": response})
            if "yes" in response.lower():
                if action == "Hesitate":
                    personas.pop(personas.index(persona))
                return persona
        if not personas:
            return None
        persona = random.choice(personas)
        if action == "Hesitate":
            personas.pop(personas.index(persona))
        return persona

    def receive(self, response):
        self.context.append(response)

    def get_engage_instruction(self):
        if self.engagement == 1:
            return "Stay vague; don't commit to the counselor's topic."
        elif self.engagement == 2:
            return f"Nod to {self.engagemented_topics[2]}; steer toward {self.engagemented_topics[1]}."
        elif self.engagement == 3:
            return f"Engage {self.engagemented_topics[1]}; hint the core issue is {self.engagemented_topics[0]}."
        elif self.engagement == 4:
            return f"Show the counselor hit your angle: {self.engagemented_topics[0]}. {self.motivation}"

    def reply(self):
        engagement_analysis = self.update_state()
        information = None
        if self.state == "Motivation":
            engage_instruction = f"Affirm they're on track re {self.engagemented_topics[0]}."
            instruction = f"[{self.motivation} {self.action2prompt['Acknowledge']} {engage_instruction}]"
            output_instruction = f"[Engage: {engage_instruction} || Motivation: {self.motivation} || Action: Acknowledge]"
            self.state = "Contemplation"
            action = "Acknowledge"
        elif self.state == "Precontemplation":
            engage_instruction = self.get_engage_instruction()
            if self.error_topic_count >= 5:
                action = "Terminate"
            else:
                action = self.select_action()
            if action == "Inform" or action == "Downplay" or action == "Blame":
                information = self.select_information(action)
                instruction = f"[{engage_instruction} {self.state2prompt[self.state]} {self.action2prompt[action]} Persona: {information}]"
                output_instruction = f"[Engage: {engagement_analysis} | {engage_instruction} | State: {self.state2prompt[self.state]} | Info: {information} | Action: {action}]"
            else:
                instruction = f"[{engage_instruction} {self.state2prompt[self.state]} {self.action2prompt[action]}]"
                output_instruction = f"[Engage: {engagement_analysis} | {engage_instruction} | State: {self.state2prompt[self.state]} | Action: {action}]"
        elif self.state == "Contemplation":
            action = self.select_action()
            if action == "Hesitate" or action == "Inform":
                information = self.select_information(action)
                instruction = f"[{self.state2prompt[self.state]} {self.action2prompt[action]} Persona: {information}]"
                output_instruction = f"[State: {self.state2prompt[self.state]} | Info: {information} | Action: {action}]"
            else:
                instruction = f"[{self.state2prompt[self.state]} {self.action2prompt[action]}]"
                output_instruction = f"[State: {self.state2prompt[self.state]} | Action: {action}]"
        else:
            if len(self.acceptable_plans) == 0:
                action = "Terminate"
            else:
                action = self.select_action()
            if action == "Inform":
                information = self.select_information(action)
                instruction = f"[{self.state2prompt[self.state]} {self.action2prompt[action]} Persona: {information}]"
                output_instruction = f"[State: {self.state2prompt[self.state]} | Info: {information} | Action: {action}]"
            elif action == "Plan":
                information = self.acceptable_plans.pop(0)
                instruction = f"[{self.state2prompt[self.state]} Plan: {information} {self.action2prompt[action]}]"
                output_instruction = f"[State: {self.state2prompt[self.state]} | Plan: {information} | Action: Plan]"
            else:
                instruction = f"[{self.state2prompt[self.state]} {self.action2prompt[action]}]"
                output_instruction = f"[State: {self.state2prompt[self.state]} | Action: {action}]"
        instruction = instruction.replace("\n", " ")
        output_instruction = output_instruction.replace("\n", " ")
        self.messages.append(
            {"role": "user", "content": f"{self.context[-1]} {instruction}"}
        )
        response = get_chatbot_response(self.messages, model=self.model)
        if not response.startswith("Client: "):
            response = f"Client: {response}"
        response = response.replace("\n", " ").strip().lstrip()
        if "Counselor: " in response:
            response = response.split("Counselor: ")[0]
        self.messages.pop(-1)
        self.messages.append({"role": "user", "content": self.context[-1]})
        self.context.append(response)
        self.messages.append({"role": "assistant", "content": response})
        return f"{output_instruction} {response}"
