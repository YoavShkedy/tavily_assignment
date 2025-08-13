# Tavily Assignment: Advanced Text Summarization System

This project implements and evaluates three different text summarization approaches using advanced NLP techniques. The system processes web content and generates summaries with varying levels of sophistication, from extractive LexRank-based summarization to complex multi-stage generative approaches.

## 🎯 Project Overview

The project compares three summarization strategies:

1. **Lite Summary**: Fast extractive summarization using LexRank algorithm
2. **Balanced Summary**: Hybrid approach combining LexRank with lightweight LLM refinement  
3. **Advanced Summary**: Multi-stage recursive summarization with premium LLMs

Each approach is evaluated across multiple dimensions including quality (G-Eval), similarity to baseline summaries (ROUGE/BERTScore), cost, and latency.

## 📁 Project Structure

```
tavily_assignment/
├── README.md                    # This file
├── summary.ipynb               # Main Jupyter notebook with implementation and analysis
├── graph_rank_summarizer.py    # LexRank implementation and configuration
├── preprocess.py               # Text cleaning and preprocessing utilities
├── summaries_1k.json          # Input dataset (1000 web documents)
├── summaries/                  # Generated summary datasets
│   ├── lite_summaries.pkl     # Lite summarization results
│   ├── balanced_summaries.pkl # Balanced summarization results
│   └── advanced_summaries.pkl # Advanced summarization results
├── g_eval_prompts/            # Evaluation prompt templates
│   ├── coherence.txt
│   ├── consistency.txt
│   ├── fluency.txt
│   └── relevance.txt
└── outline.pdf                # Project specification document
```

## 🔧 Core Modules

### GraphRankSummarizer (`graph_rank_summarizer.py`)

Implements the **LexRank algorithm** for extractive text summarization:

- **`LexRankConfig`**: Configuration class for summarization parameters
- **`LexRankSummarizer`**: Main summarization engine using PageRank on sentence similarity graphs
- **Key Features**:
  - TF-IDF vectorization with configurable n-gram ranges
  - Cosine similarity-based sentence graph construction
  - PageRank scoring for sentence importance ranking
  - Redundancy filtering using Jaccard similarity
  - Configurable output length and quality thresholds

### Preprocessing (`preprocess.py`)

Provides **web content cleaning utilities**:

- **`clean_web_text()`**: Primary function for cleaning raw web content
- **Features**:
  - Markdown link and image removal
  - Table and navigation content filtering
  - Whitespace normalization
  - Link density-based content filtering
  - Bullet point and boilerplate removal

## 📊 Implementation Details

### Lite Summary (Extractive)
- **Algorithm**: LexRank with TF-IDF similarity
- **Speed**: ~0.083s average, 12 summaries/second
- **Cost**: No API costs (local computation)
- **Quality**: High factual consistency, lower coherence

### Balanced Summary (Hybrid)
- **Pipeline**: LexRank extraction → Amazon Nova Micro refinement
- **Speed**: ~1.56s average, 0.64 summaries/second  
- **Cost**: ~$0.027 per 1,000 summaries
- **Quality**: Best overall G-Eval scores

### Advanced Summary (Multi-stage)
- **Pipeline**: Recursive Nova Lite compression → Claude Sonnet 4 polish
- **Speed**: ~26.8s average, 0.04 summaries/second
- **Cost**: ~$10.21 per 1,000 summaries
- **Quality**: Surprisingly lower than expected due to error accumulation

## 🔑 Environment Setup

**⚠️ Important Note**: The `.env` file is **not provided** in this repository as it contains sensitive API keys for AWS Bedrock services. You'll need to create your own `.env` file with the following structure:

```bash
# .env file (create this yourself)
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1
```

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd tavily_assignment
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Key dependencies include:
   - `scikit-learn` (TF-IDF vectorization)
   - `numpy` (numerical operations)
   - `boto3` (AWS Bedrock integration)
   - `langchain-aws` (LLM orchestration)
   - `rouge-score` (evaluation metrics)
   - `bert-score` (semantic evaluation)
   - `python-dotenv` (environment management)
   - `tqdm` (progress tracking)

3. **Configure AWS credentials**:
   - Create a `.env` file with your AWS Bedrock API credentials
   - Ensure you have access to the required models:
     - `amazon.nova-micro-v1:0`
     - `amazon.nova-lite-v1:0` 
     - `us.anthropic.claude-sonnet-4-20250514-v1:0`

4. **Launch Jupyter**:
   ```bash
   jupyter notebook summary.ipynb
   ```

## 📈 Usage

The main analysis is contained in `summary.ipynb`, which demonstrates:

1. **Data Processing**: Loading and preprocessing the 1,000-document dataset
2. **Lite Summarization**: Extractive summarization with performance analysis
3. **Balanced Summarization**: Hybrid approach with cost-quality optimization
4. **Advanced Summarization**: Multi-stage generative summarization
5. **Comprehensive Evaluation**: G-Eval quality assessment and traditional metrics

### Running Individual Components

```python
# Use the preprocessing module
from preprocess import clean_web_text
cleaned_text = clean_web_text(raw_web_content)

# Use the GraphRankSummarizer
from graph_rank_summarizer import LexRankSummarizer, LexRankConfig

config = LexRankConfig(max_chars=1200, threshold=0.1)
summarizer = LexRankSummarizer(config)
result = summarizer.summarize(title=None, text=cleaned_text)
summary = result['summary']
```

## 🧪 Evaluation Results

### G-Eval Quality Scores (1-5 scale, higher = better)

| Method | Coherence | Consistency | Fluency | Relevance |
|--------|-----------|-------------|---------|-----------|
| **Balanced** | **3.785** | 4.371 | **2.924** | **4.004** |
| **Lite** | 1.940 | **4.706** | 1.662 | 2.265 |
| **Advanced** | 2.693 | 2.449 | 2.933 | 2.600 |

### Performance Metrics

| Method | Avg Time | Cost/1K | ROUGE-1 F1 | BERTScore F1 |
|--------|----------|---------|-------------|--------------|
| **Lite** | 0.083s | $0.00 | **0.386** | **0.790** |
| **Balanced** | 1.56s | $0.027 | 0.305 | 0.758 |
| **Advanced** | 26.8s | $10.21 | 0.298 | 0.757 |

## 💡 Key Findings

1. **Balanced approach optimal**: Best trade-off between quality, speed, and cost
2. **Complexity ≠ Better performance**: Advanced multi-stage pipeline underperformed 
3. **Evaluation paradigm matters**: G-Eval (quality) vs ROUGE (similarity) yield different winners
4. **Extractive preserves fidelity**: Lite summaries excel at factual consistency
