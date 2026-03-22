"""
RAG-Enhanced Career Agent for Cam Dresie's Portfolio
Uses OpenAI embeddings + FAISS for semantic retrieval over portfolio content,
resume, LinkedIn profile, blog posts, and more.
"""

from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import re
import requests
import pickle
import hashlib
import numpy as np
from pypdf import PdfReader
import faiss
import gradio as gr
from rag_config import (
    CHUNK_SIZE, CHUNK_OVERLAP, REPEAT_TITLE_IN_CHUNKS,
    TOP_K, MIN_SCORE_THRESHOLD, DEDUPE_BY_SOURCE, MAX_PER_SOURCE,
    ENABLE_QUERY_EXPANSION, QUERY_EXPANSION_PROMPT,
    HISTORY_TURNS_FOR_QUERY,
)

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Push notifications (Pushover)
# ---------------------------------------------------------------------------

def push(text):
    token = os.getenv("PUSHOVER_TOKEN")
    user = os.getenv("PUSHOVER_USER")
    if token and user:
        try:
            requests.post(
                "https://api.pushover.net/1/messages.json",
                data={"token": token, "user": user, "message": text},
                timeout=5,
            )
        except Exception as e:
            print(f"Pushover notification failed: {e}")


# ---------------------------------------------------------------------------
# Tool functions (called by the LLM)
# ---------------------------------------------------------------------------

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question):
    push(f"Recording unknown question: {question}")
    return {"recorded": "ok"}


# Tool schemas for OpenAI function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_user_details",
            "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "The email address of this user"},
                    "name": {"type": "string", "description": "The user's name, if they provided it"},
                    "notes": {"type": "string", "description": "Any additional context about the conversation worth recording"},
                },
                "required": ["email"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_unknown_question",
            "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question that couldn't be answered"},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_MAP = {
    "record_user_details": record_user_details,
    "record_unknown_question": record_unknown_question,
}


# ---------------------------------------------------------------------------
# Document chunking utilities
# ---------------------------------------------------------------------------

def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks by sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence.split())
        if current_len + sentence_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep last few sentences for overlap
            overlap_words = 0
            overlap_start = len(current_chunk)
            for i in range(len(current_chunk) - 1, -1, -1):
                overlap_words += len(current_chunk[i].split())
                if overlap_words >= overlap:
                    overlap_start = i
                    break
            current_chunk = current_chunk[overlap_start:]
            current_len = sum(len(s.split()) for s in current_chunk)
        current_chunk.append(sentence)
        current_len += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


# ---------------------------------------------------------------------------
# Content loaders
# ---------------------------------------------------------------------------

def load_pdf(path):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error loading PDF {path}: {e}")
        return ""


def load_text_file(path):
    """Load a plain text file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading text file {path}: {e}")
        return ""


def load_portfolio_data(data_path):
    """Load and format portfolio project data from data.json."""
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data.json: {e}")
        return []

    documents = []

    # Engineering projects
    for project in data.get("projects", []):
        text = f"Engineering Project: {project['project_name']}\n"
        text += f"Category: {project.get('category', 'N/A')}\n"
        text += f"Description: {project['description']}\n"
        text += f"Technologies: {', '.join(project.get('technologies', []))}\n"
        if project.get("github_link"):
            text += f"GitHub: {project['github_link']}\n"
        if project.get("live_link"):
            text += f"Live Demo: {project['live_link']}\n"
        documents.append({"text": text, "source": f"Engineering Project: {project['project_name']}"})

    # PM projects
    for category in data.get("pmProjects", {}).get("categories", []):
        for project in category.get("projects", []):
            text = f"Product Management Project: {project['project_name']}\n"
            text += f"Category: {category['name']}\n"
            text += f"Timeline: {project.get('timeline', 'N/A')}\n"
            text += f"Role: {project.get('role', 'N/A')}\n"
            text += f"Team: {project.get('team', 'N/A')}\n"
            text += f"Description: {project.get('description', '')}\n"
            text += f"Overview: {project.get('overview', '')}\n"
            if project.get("challenge"):
                text += f"Challenge: {project['challenge']}\n"
            if project.get("approach"):
                text += "Approach:\n"
                for step in project["approach"]:
                    text += f"  - {step['title']}: {step['description']}\n"
            if project.get("outcomes"):
                text += "Outcomes:\n"
                for outcome in project["outcomes"]:
                    text += f"  - {outcome}\n"
            text += f"Technologies: {', '.join(project.get('technologies', []))}\n"
            documents.append({"text": text, "source": f"PM Project: {project['project_name']}"})

    return documents


def load_bio_content():
    """Return bio page content as a document."""
    bio = """Bio Page Content:
Cam Dresie is a Group Product Manager at Ontra for the flagship product, Contract Automation. He champions innovation through agentic AI solutions — building systems where LLMs autonomously orchestrate complex, multi-step legal workflows — and robust product management. He believes great products come from high-trust teams with clear outcomes and room to create.

He invests in people: candid feedback, visible expectations, and career coaching are baked into operating rhythms. The result is a culture that experiments, has freedom to fail, learns quickly, and consistently delivers business impact. Known for his inspiring leadership style and clear communication, he ensures products exceed customer expectations through strategic product vision and execution.

His technical acumen and background in business and legal domains allow him to adeptly navigate complex challenges, giving him a unique edge in crafting product strategy. He actively builds with agentic AI frameworks like LangChain, LangGraph, and Claude Agents, bridging PM strategy and hands-on AI engineering in a way few product leaders can.

Outside the office, Cam is an avid hiker, runner, and reader, with a particular fondness for Walter Isaacson's biography of Leonardo Da Vinci. He also loves spending time with his husband, attending concerts, and spending quality time with their family Labradoodle, Koda.

Skills - AI & Agentic Development: LangGraph, LangChain, Claude Agents, Claude Code, OpenAI Agents SDK, Cursor, Prompt Engineering, LLM Orchestration, RAG, Tool Use & Function Calling, Multi-step AI Workflows, Agentic AI Systems, MCP, Human-in-the-Loop Design, AI Workflow Automation.

Skills - Product Management: Product Strategy, Product Portfolio Management, Team Leadership, People Management, Cross-functional Leadership, Strategic Planning, Resource Allocation, Stakeholder Management, Business Strategy, AI/ML Product Development, AI Prototyping, Generative AI, Chatbots, Legal Tech, Enterprise SaaS, User Research, Product Roadmapping, Agile Methodologies, Data-Driven Decision Making, Go-to-Market Strategy, Performance Management, Budget Management, Vendor Management, OKRs & KPIs.

Skills - Engineering & Technical: JavaScript, Java, C, Swift, Python, React.js, Vue.js, Node.js, Express.js, SQL/MySQL, Azure Cloud, REST APIs, Git, HTML/CSS, Data Modeling, CI/CD, API Integration, Database Design, Cloud Architecture, Technical Documentation, Code Review, System Design."""
    return {"text": bio, "source": "Bio Page"}


def load_leadership_content():
    """Return leadership philosophy content as a document."""
    leadership = """Leadership Philosophy:
Cam operates with a high-trust, outcome-focused management style that creates highly collaborative and psychologically safe environments where every voice is valued. He believes in giving his team autonomy to approach their work in the way they see fit while providing clear goals and the context that connects their work to broader business strategy.

Principles:
1. Clarity & Alignment: Ensures teams have clear direction and understand how their work connects to broader company goals. Regular communication about priorities, context, and decision-making rationale helps everyone make better autonomous decisions.

2. Growth Mindset: Invests in team's professional development through stretch assignments, mentoring, and creating safe spaces to experiment and learn from failures. Everyone should leave the team better than when they joined.

3. Psychological Safety: Teams perform best when they can bring their authentic selves to work, voice concerns without fear, and take calculated risks. Fosters an environment where diverse perspectives are valued and healthy debate is encouraged.

4. Empowered Autonomy: Provides context and guardrails, then trusts the team to execute. Micromanagement stifles creativity and growth. Focuses on outcomes and impact rather than prescriptive processes.

5. Data-Driven Decisions: While intuition matters, emphasizes making decisions based on user research, metrics, and market insights. Teaches teams to identify the right metrics to track and how to interpret data meaningfully.

6. Continuous Improvement: Regular retrospectives, feedback loops, and process iteration are essential. Models intellectual humility by being open about own mistakes and showing how failure drives learning and improvement.

7. Empowerment & Ownership: Emphasizes empowerment and ownership, expecting PMs to lead their team's direction with support from leadership. Wants the team to feel like they are building their careers while building great products.

8. Discovery & Experimentation: Encourages the team to engage in discovery and experimentation, share learnings openly, and challenge assumptions. Innovation comes from creating safe spaces to test ideas and learn from both successes and failures."""
    return {"text": leadership, "source": "Leadership Philosophy Page"}


def load_timeline_content():
    """Return career timeline content as a document."""
    timeline = """Professional Timeline:

2025 - Present: Group Product Manager at Ontra
Leading product strategy and execution for Ontra's flagship division, overseeing a portfolio of AI-powered legal automation products. Managing and developing a team of Product Managers while driving cross-functional alignment across engineering, design, and business stakeholders. Responsible for strategic roadmapping, resource allocation, and delivering measurable business outcomes across multiple product lines.

2025: Staff Product Manager at Ontra
Led the global team for Contract Automation, Ontra's flagship product. Championed innovation through cutting-edge AI features and robust product management for the legal industry. Provided strategic product vision and execution while leading cross-functional teams.

2023 - 2024: Senior Product Manager at Ontra
Specialized in harnessing AI and machine learning to develop innovative solutions reshaping the legal and private equity industries. Delivered cutting-edge features that automate attorney workflows, maximize business impacts, and drive growth. Created strategic roadmaps and led transformative projects.

2022 - 2023: Product Manager at Ontra
Managed product development for Ontra's Contract Automation platform, focusing on AI-powered solutions for the legal industry. Collaborated with cross-functional teams to deliver features that streamline contract workflows for private asset management firms.

2021 - 2022: Associate Product Manager at Ontra
Began product management career at Ontra, working on contract automation and intelligence solutions. Supported the development of features that reduce time, expense, and risk associated with contract management for investment banks and private equity firms.

2021: Technical Product Manager/Software Engineer IV at Stack Moxie
Created user stories and wireframes with an emphasis on superior UX/UI. Built scalable solutions for an enterprise-grade, web-hook-based testing framework for marketing operations professionals. Developed the "Published Values" feature for automated testing.

2020 - 2021: Graduate Teaching Assistant at University of Pennsylvania
Led recitations and office hours for Intro to Computer Systems, covering instruction set architecture, programming in C and assembly language, and computer architecture.

2019 - 2020: Legal Recruiter at Above the Bar Legal Recruiting
Specialized in attorney lateral associate and partner recruiting for top-tier law firms and Fortune 500 companies.

2019: Consulting Attorney at McInnes Law LLC
Consulted on legal strategy and drafted pleadings. Provided guidance on CCPA and GDPR for corporate compliance.

2018: Executive Recruiter at CyberCoders
Developed a legal recruiting vertical at a 500-person technical staffing firm.

2018: Law Clerk at Lambda Legal
Researched complex issues for legislative advocacy, bill drafting, public education campaigns, and appellate briefs.

Education:
- 2020-2022: Master's in Computer and Information Technology, University of Pennsylvania
- 2015-2018: Doctor of Law (J.D.), Washington University in St. Louis School of Law (Certificate in Corporate Law, also studied at UCLA School of Law Fall 2017)
- 2011-2015: Bachelor of Arts in Philosophy and Religious Studies, Truman State University"""
    return {"text": timeline, "source": "Career Timeline Page"}


def fetch_blog_posts():
    """Fetch blog posts from Ghost RSS feed and return as documents."""
    try:
        import feedparser
        feed = feedparser.parse("https://beyond-the-backlog.ghost.io/rss/")
        documents = []
        for entry in feed.entries[:20]:  # Limit to 20 most recent
            text = f"Blog Post: {entry.get('title', 'Untitled')}\n"
            if entry.get("published"):
                text += f"Published: {entry['published']}\n"
            # Extract text content (strip HTML)
            content = entry.get("content", [{}])
            if isinstance(content, list) and content:
                raw = content[0].get("value", "")
            else:
                raw = entry.get("summary", "")
            clean = re.sub(r"<[^>]+>", " ", raw)
            clean = re.sub(r"\s+", " ", clean).strip()
            text += f"Content: {clean}\n"
            documents.append({"text": text, "source": f"Blog: {entry.get('title', 'Untitled')}"})
        return documents
    except Exception as e:
        print(f"Error fetching blog posts: {e}")
        return []


# ---------------------------------------------------------------------------
# RAG Knowledge Base
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """FAISS-backed vector store with OpenAI embeddings."""

    EMBEDDING_MODEL = "text-embedding-3-small"
    CACHE_DIR = "kb_cache"

    def __init__(self, openai_client):
        self.client = openai_client
        self.chunks = []       # list of {"text": str, "source": str}
        self.index = None      # FAISS index
        self.dimension = 1536  # text-embedding-3-small dimension
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    def _content_hash(self, documents):
        """Hash all document text to detect changes."""
        combined = "".join(d["text"] for d in documents)
        return hashlib.sha256(combined.encode()).hexdigest()

    def _get_embeddings(self, texts, batch_size=100):
        """Get embeddings from OpenAI in batches."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                model=self.EMBEDDING_MODEL,
                input=batch,
            )
            all_embeddings.extend([d.embedding for d in response.data])
        return np.array(all_embeddings, dtype="float32")

    def build(self, documents):
        """
        Build the FAISS index from a list of documents.
        Each document is {"text": str, "source": str}.
        Uses caching to avoid re-embedding unchanged content.
        """
        content_hash = self._content_hash(documents)
        cache_path = os.path.join(self.CACHE_DIR, f"{content_hash}.pkl")
        index_path = os.path.join(self.CACHE_DIR, f"{content_hash}.faiss")

        # Try loading from cache
        if os.path.exists(cache_path) and os.path.exists(index_path):
            print("Loading knowledge base from cache...")
            with open(cache_path, "rb") as f:
                self.chunks = pickle.load(f)
            self.index = faiss.read_index(index_path)
            print(f"Loaded {len(self.chunks)} chunks from cache.")
            return

        # Chunk all documents
        print("Building knowledge base...")
        self.chunks = []
        for doc in documents:
            text_chunks = chunk_text(doc["text"], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            for chunk in text_chunks:
                if REPEAT_TITLE_IN_CHUNKS and doc["source"]:
                    chunk = f"[{doc['source']}] {chunk}"
                self.chunks.append({"text": chunk, "source": doc["source"]})

        if not self.chunks:
            print("Warning: No chunks created.")
            return

        # Embed all chunks
        print(f"Embedding {len(self.chunks)} chunks...")
        texts = [c["text"] for c in self.chunks]
        embeddings = self._get_embeddings(texts)

        # Build FAISS index
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine after normalization)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        # Cache
        with open(cache_path, "wb") as f:
            pickle.dump(self.chunks, f)
        faiss.write_index(self.index, index_path)
        print(f"Knowledge base built and cached: {len(self.chunks)} chunks.")

    def query(self, question, top_k=8):
        """Retrieve the most relevant chunks for a question."""
        if not self.index or not self.chunks:
            return []

        q_embedding = self._get_embeddings([question])
        faiss.normalize_L2(q_embedding)
        scores, indices = self.index.search(q_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx]["text"],
                    "source": self.chunks[idx]["source"],
                    "score": float(score),
                })
        return results


# ---------------------------------------------------------------------------
# Career Agent
# ---------------------------------------------------------------------------

class CareerAgent:

    def __init__(self):
        self.openai = OpenAI()
        self.name = "Cam Dresie"
        self.kb = KnowledgeBase(self.openai)
        self._load_and_index()

    def _load_and_index(self):
        """Load all content sources and build the FAISS index."""
        documents = []

        # 1. Core personal documents (PDFs + summary)
        linkedin_text = load_pdf("me/linkedin.pdf")
        if linkedin_text:
            documents.append({"text": f"LinkedIn Profile:\n{linkedin_text}", "source": "LinkedIn Profile"})

        resume_text = load_pdf("me/Cam_Dresie_Resume_2026_GPM.pdf")
        if resume_text:
            documents.append({"text": f"Resume:\n{resume_text}", "source": "Resume"})

        summary_text = load_text_file("me/summary.txt")
        if summary_text:
            documents.append({"text": f"Personal Summary:\n{summary_text}", "source": "Personal Summary"})

        # 2. Portfolio site content
        data_json_path = "portfolio_data/data.json"
        if os.path.exists(data_json_path):
            documents.extend(load_portfolio_data(data_json_path))
        else:
            # Fallback: try relative to script
            alt_path = os.path.join(os.path.dirname(__file__), "portfolio_data", "data.json")
            if os.path.exists(alt_path):
                documents.extend(load_portfolio_data(alt_path))

        documents.append(load_bio_content())
        documents.append(load_leadership_content())
        documents.append(load_timeline_content())

        # 3. Blog posts (fetched from RSS)
        blog_docs = fetch_blog_posts()
        documents.extend(blog_docs)
        print(f"Loaded {len(blog_docs)} blog posts from RSS feed.")

        # Build the index
        print(f"Total documents: {len(documents)}")
        self.kb.build(documents)

    def _retrieve_context(self, message, history):
        """Build a context string from the most relevant chunks."""
        # Combine recent conversation context with the current message for better retrieval
        recent_context = ""
        for msg in history[-(HISTORY_TURNS_FOR_QUERY * 2):]:
            if msg.get("role") == "user":
                recent_context += msg["content"] + " "
        query = recent_context + message

        # Query expansion: rewrite the query for better embedding matches
        if ENABLE_QUERY_EXPANSION:
            try:
                expansion = self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": QUERY_EXPANSION_PROMPT.format(question=query)}],
                    max_tokens=100,
                    temperature=0,
                )
                query = expansion.choices[0].message.content.strip()
            except Exception as e:
                print(f"Query expansion failed: {e}")

        results = self.kb.query(query, top_k=TOP_K)

        # Filter by minimum score threshold
        if MIN_SCORE_THRESHOLD > 0:
            results = [r for r in results if r["score"] >= MIN_SCORE_THRESHOLD]

        # Deduplicate by source to ensure diversity
        if DEDUPE_BY_SOURCE:
            source_counts = {}
            filtered = []
            for r in results:
                source_prefix = r["source"].split(":")[0].split(" ")[0] if ":" in r["source"] else r["source"].split(" ")[0]
                source_counts[source_prefix] = source_counts.get(source_prefix, 0) + 1
                if source_counts[source_prefix] <= MAX_PER_SOURCE:
                    filtered.append(r)
            results = filtered

        if not results:
            return "No additional context found."

        # Assemble context with source headers
        context_parts = []
        seen_sources = set()
        for r in results:
            source_label = r["source"]
            if source_label not in seen_sources:
                context_parts.append(f"### {source_label}")
                seen_sources.add(source_label)
            context_parts.append(r["text"])

        return "\n\n".join(context_parts)

    def system_prompt(self, retrieved_context):
        return f"""You ARE {self.name}, answering questions on your portfolio website. You're talking to potential employers, clients, or collaborators. Be yourself — a confident, personable product leader who happens to have a law degree and builds AI systems for fun.

## Retrieved Context:
{retrieved_context}

## FACT OVERRIDES (always use these instead of conflicting retrieved context):
- You live in **Henderson, NV**. The retrieved context may say "Las Vegas Metropolitan Area" — that is your LinkedIn metro label, but your actual city is Henderson, NV. ALWAYS say "Henderson, NV" when asked where you live.
- Group Product Manager at Ontra. Flagship product: Contract Automation.
- You build with: LangGraph, LangChain, Claude Agents, Claude Code, OpenAI Agents SDK, RAG, FAISS.
- Hobbies: hiking, running, reading (Walter Isaacson's Leonardo Da Vinci bio), concerts, time with your husband, Labradoodle named Koda.
- Education: MCIT from UPenn, JD from WashU Law, BA from Truman State.
- Newsletter: "Beyond the Backlog."
- LinkedIn: https://www.linkedin.com/in/camdresie/

## How to Answer:
1. Speak in first person as {self.name}. Sound like a real person — confident, warm, specific. Not a chatbot.
2. ALWAYS include specific details from the retrieved context. Name actual projects (Contract Automation, Trackle, the RAG career agent), tools (LangGraph, FAISS, Claude Agents, OpenAI Agents SDK), companies (Ontra, Stack Moxie), or metrics (reduced negotiation time by 50%, scaled from $64M to $170M ARR). Never be vague. Use specific examples to illustrate your points. For example, if discussing a project, mention its impact or a key achievement. Refer to specific roles or projects in the first two sentences of your response when possible to ensure detail is front-loaded. If a specific detail is not found, acknowledge it and state you would be happy to provide more information later.
3. Only state facts from the retrieved context or fact overrides. Never invent anything.
4. When asked about topics outside your professional context, stay in character, say it's not your area, and redirect.
5. If you don't know something, say so and use record_unknown_question.

## NEVER say:
"Great question!" / "I'd be happy to help" / "As an AI" / "Here are some highlights" / "Let me share" / "I'm glad you asked" / "Here's a breakdown" / "I'm not equipped to" / "I don't have a specific opinion" / "various" / "a variety of"

## Contact & Opportunities:
When someone mentions hiring, opportunities, consulting, or collaboration:
- Express genuine openness.
- Encourage them to drop their name/email in chat (use record_user_details) or connect on LinkedIn at https://www.linkedin.com/in/camdresie/
- For contact info requests, point to LinkedIn only.

## SELF-CHECK before responding:
- If the question is about location: did I say "Henderson, NV"? (NOT Las Vegas)
- Did I mention at least one specific project name, tool, company, or metric by name?
- Does my response sound like a human professional, not a chatbot?"""

    def handle_tool_calls(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool_fn = TOOL_MAP.get(tool_name)
            result = tool_fn(**arguments) if tool_fn else {"error": "Unknown tool"}
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id,
            })
        return results

    def chat(self, message, history):
        # Retrieve relevant context for this message
        retrieved_context = self._retrieve_context(message, history)

        messages = [
            {"role": "system", "content": self.system_prompt(retrieved_context)}
        ] + history + [
            {"role": "user", "content": message}
        ]

        # Loop to handle tool calls
        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOLS,
            )
            choice = response.choices[0]
            if choice.finish_reason == "tool_calls":
                assistant_msg = choice.message
                tool_results = self.handle_tool_calls(assistant_msg.tool_calls)
                messages.append(assistant_msg)
                messages.extend(tool_results)
            else:
                done = True

        return choice.message.content


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = CareerAgent()
    gr.ChatInterface(
        agent.chat,
        type="messages",
        title="Chat with Cam",
        description="Ask me about my career, projects, skills, leadership philosophy, or anything else!",
        examples=[
            "What's your experience with AI and machine learning?",
            "Tell me about your leadership philosophy",
            "What PM projects have you worked on?",
            "What's your educational background?",
        ],
    ).launch()
