"""Configuration constants for the newsletter agent."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Timeouts & Rate Limits ---
REQUEST_TIMEOUT = 900  # 15 minutes
SHORT_REQUEST_TIMEOUT = 60
DOMAIN_RATE_LIMIT = 5.0  # seconds between requests to same domain

# --- Concurrency ---
DEFAULT_CONCURRENCY = 12
MAX_CRITIQUE_ITERATIONS = 2

# --- File Paths ---
DOWNLOAD_ROOT = "download"
DOWNLOAD_DIR = os.path.join(DOWNLOAD_ROOT, "sources")
PAGES_DIR = os.path.join(DOWNLOAD_ROOT, "html")
TEXT_DIR = os.path.join(DOWNLOAD_ROOT, "text")
SCREENSHOT_DIR = os.path.join(DOWNLOAD_ROOT, "screenshots")

# --- Database ---
NEWSAGENTDB = "newsletter_agent.db"

# --- Browser ---
FIREFOX_PROFILE_PATH = os.environ.get("FIREFOX_PROFILE_PATH", "")

# --- Content Filtering ---
MIN_TITLE_LEN = 28
DOMAIN_SKIPLIST = ["finbold.com", "philarchive.org"]
IGNORE_LIST = [
    "www.bloomberg.com", "bloomberg.com",
    "cnn.com", "www.cnn.com",
    "wsj.com", "www.wsj.com",
]

# --- Model Constants ---
CLAUDE_SONNET = "claude-sonnet-4-5-20250929"
CLAUDE_HAIKU = "claude-haiku-4-5-20251001"
DEFAULT_MODEL = CLAUDE_SONNET

# --- Canonical Topics (verbatim from original, active entries only) ---
CANONICAL_TOPICS = [
    "Policy And Regulation",
    "Economics",
    "Exports And Trade",
    "Governance",
    "Safety And Alignment",
    "Bias And Fairness",
    "Privacy And Surveillance",
    "Inequality",
    "Automation",
    "Disinformation",
    "Deepfakes",
    "Sustainability",

    "Agents",
    "Coding Assistants",

    "Virtual Assistants",
    "Chatbots",
    "Robots",
    "Autonomous Vehicles",
    "Drones",
    "Virtual And Augmented Reality",

    "Reinforcement Learning",
    "Language Models",
    "Transformers",
    "Gen AI",
    "Retrieval Augmented Generation",
    "Computer Vision",
    "Facial Recognition",
    "Speech Recognition And Synthesis",

    "Open Source",

    "Internet of Things",
    "Quantum Computing",
    "Brain-Computer Interfaces",

    "Hardware",
    "Infrastructure",
    "Data Centers",
    "Enterprise AI",
    "Hyperscalers",
    "Adoption and Spending",
    "Semiconductor Chips",
    "GPUs",
    "AI Compute",
    "Neuromorphic Computing",

    "Healthcare",
    "Mental Health",
    "Fintech",
    "Education",
    "Entertainment",
    "Funding",
    "Venture Capital",
    "Mergers and Acquisitions",
    "Deals",
    "IPOs",
    "Ethics",
    "Legal Issues",
    "Cybersecurity",
    "AI Doom",
    "Stocks",
    "Valuation Bubble",
    "Cryptocurrency",
    "Climate",
    "Energy",
    "Nuclear",
    "Scams",
    "Intellectual Property",
    "Customer Service",
    "Military",
    "Agriculture",
    "Testing",
    "Authors And Writing",
    "Books And Publishing",
    "TV And Film And Movies",
    "Streaming",
    "Hollywood",
    "Music",
    "Art And Design",
    "Fashion",
    "Food And Drink",
    "Travel",
    "Health And Fitness",
    "Sports",
    "Gaming",
    "Politics",
    "Finance",
    "History",
    "Society And Culture",
    "Lifestyle And Travel",
    "Jobs And Careers",
    "Labor Markets And Productivity",
    "Products",
    "Opinion",
    "Review",
    "Cognitive Science",
    "Consciousness",
    "Artificial General Intelligence",
    "Singularity",
    "Manufacturing",
    "Supply Chain Optimization",
    "Transportation",
    "Smart Grid",
    "Recommendation Systems",

    "Nvidia",
    "Google",
    "OpenAI",
    "Meta",
    "xAI",
    "Perplexity",
    "Anthropic",
    "Tesla",

    "ChatGPT",
    "Gemini",
    "Claude",
    "Copilot",
    "Grok",

    "Elon Musk",
    "Sam Altman",
    "Mustafa Suleyman",
    "Demis Hassabis",
    "Jensen Huang",

    "China",
    "European Union",
    "UK",
    "Russia",
    "Japan",
    "India",
    "Korea",
    "Taiwan",
]
