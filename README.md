# 🔎 Multi-Agent Research Assistant

A fully autonomous AI research system powered by multiple specialized agents that collaborate to produce professional research reports on any topic.

## 🚀 Live Demo
**Try it here → [Multi-Agent Research Assistant](https://multi-agent-research-systemm.streamlit.app/)**

---

## 🤖 What It Does

```
User gives a topic
      ↓
🔍 Search Agent    → searches live web via Tavily
      ↓
📖 Reader Agent    → scrapes & extracts content from sources
      ↓
✍️ Writer Chain    → writes a professional research report
      ↓
🧐 Critic Chain    → reviews & scores the report
      ↓
📊 NLP Analysis    → extracts keywords & sentiment
      ↓
🤖 BERT Sentiment  → Deep Learning sentiment analysis
      ↓
⭐ ML Scoring      → predicts report quality score
      ↓
📄 RAG (optional)  → combine web + your uploaded PDFs
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Agentic AI | LangChain + Groq (Llama 3.3) |
| Web Search | Tavily API |
| Web Scraping | BeautifulSoup4 |
| RAG | ChromaDB + HuggingFace Embeddings |
| NLP | NLTK + TextBlob |
| Deep Learning | HuggingFace BERT (DistilBERT) |
| Machine Learning | Scikit-learn |
| UI | Streamlit |
| CI/CD | GitHub Actions |
| Deployment | Streamlit Cloud |

---

## 📁 Project Structure

```
multi-agent-research-system/
├── .github/
│   └── workflows/
│       └── cicd.yml        ← CI/CD pipeline
├── tests/
│   └── test_tools.py       ← automated tests
├── tools.py                ← Tavily + BeautifulSoup tools
├── agents.py               ← 4 agents + 2 chains
├── pipeline.py             ← orchestrates all agents
├── rag.py                  ← PDF upload + ChromaDB
├── nlp_analyzer.py         ← keyword extraction + sentiment
├── dl_sentiment.py         ← BERT deep learning sentiment
├── ml_scorer.py            ← ML quality scoring
├── app.py                  ← Streamlit UI
└── requirements.txt
```

---

## ⚡ Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/cout1909/multi-agent-research-system.git
cd multi-agent-research-system
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Create `.env` file**
```env
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
```

**5. Run the app**
```bash
streamlit run app.py
```

---

## 🔑 Getting Free API Keys

| Service | URL | Free Tier |
|---|---|---|
| Groq | console.groq.com | ✅ Free |
| Tavily | app.tavily.com | ✅ 1000 searches/month free |

---

## ✨ Features

- ✅ **Multi-Agent System** — 4 specialized agents collaborate
- ✅ **Live Web Search** — searches real-time internet
- ✅ **RAG Support** — upload your own PDFs
- ✅ **NLP Analysis** — keyword extraction + sentiment
- ✅ **Deep Learning** — BERT sentiment analysis
- ✅ **ML Quality Scoring** — predicts report quality
- ✅ **Professional UI** — dark themed Streamlit interface
- ✅ **Live Pipeline Display** — watch agents work in real time
- ✅ **CI/CD** — automated testing with GitHub Actions

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests
- Star the repo if you find it useful ⭐

---

## 📄 License

MIT License — free to use, modify and distribute.

---

## 👨‍💻 Built By

**PRATIK BITKE** — Aspiring AI Engineer

## 👨‍💻 Built By

**Pratik Bitke** — Aspiring AI Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/pratik-bitke-bb1749271)
[![GitHub](https://img.shields.io/github/stars/cout1909/multi-agent-research-system?style=social)](https://github.com/cout1909/multi-agent-research-system)