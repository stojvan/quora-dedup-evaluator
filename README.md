# Quora Deduplication Green Agent

A green agent that evaluates purple agents on the Quora Question Pairs deduplication task. Built using the [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/) protocol and compatible with the [AgentBeats](https://agentbeats.dev) platform.

## Overview

This green agent tests a purple agent's ability to identify duplicate question pairs from the Quora Question Pairs dataset. The agent:
- Sends pairs of questions to a purple agent
- Collects binary predictions (1 = duplicate, 0 = not duplicate)
- Compares predictions against ground truth labels
- Reports comprehensive accuracy metrics

## Dataset

The agent uses the [Quora Question Pairs dataset](https://huggingface.co/datasets/quora) from Hugging Face, which contains ~400K question pairs labeled as duplicates or non-duplicates. The dataset is downloaded and cached automatically on first run.

## Project Structure

```
src/
├─ server.py       # Server setup and agent card configuration
├─ executor.py     # A2A request handling
├─ agent.py        # Core evaluation logic
├─ messenger.py    # A2A messaging utilities
├─ data_loader.py  # Dataset loading and sampling
└─ schemas.py      # Pydantic models for requests/responses
tests/
└─ test_agent.py   # Agent tests
Dockerfile         # Docker configuration
pyproject.toml     # Python dependencies
```

## Assessment Request Format

The green agent expects requests in the following format:

```json
{
  "participants": {
    "deduplication_agent": "http://purple-agent:9010"
  },
  "config": {
    "sample_size": 100,
    "random_seed": 777
  }
}
```

### Configuration Parameters

- **sample_size** (required): Number of question pairs to evaluate (1-1000)
- **random_seed** (optional): Integer seed for reproducible sampling

## Purple Agent Requirements

Purple agents being evaluated must:

1. Accept question pairs as JSON:
```json
{
  "question1": "What is the best way to learn Python?",
  "question2": "How can I learn Python effectively?"
}
```

2. Respond with a prediction (recommended format):
```json
{
  "prediction": 1,
  "justification": "Both questions ask about learning Python effectively"
}
```

Or minimal format:
```json
{
  "prediction": 0
}
```

The agent also accepts plain integer responses (`"1"` or `"0"`) for compatibility.

## Results Format

The green agent returns comprehensive evaluation metrics:

```json
{
  "assessment_type": "quora_deduplication",
  "sample_size": 100,
  "evaluated_pairs": 98,
  "accuracy": 0.8571,
  "correct_predictions": 84,
  "incorrect_predictions": 14,
  "confusion_matrix": {
    "true_positives": 42,
    "true_negatives": 42,
    "false_positives": 7,
    "false_negatives": 7
  },
  "metrics": {
    "precision": 0.8571,
    "recall": 0.8571,
    "f1_score": 0.8571
  },
  "execution_time_seconds": 45.2,
  "error_rate": 0.02,
  "errors_count": 2,
  "sample_errors": [...]
}
```

## Running Locally

```bash
# Install dependencies
uv sync

# Run the server
uv run src/server.py
```

## Running with Docker

```bash
# Build the image
docker build -t my-agent .

# Run the container
docker run -p 9009:9009 my-agent
```

## Testing

Run A2A conformance tests against your agent.

```bash
# Install test dependencies
uv sync --extra test

# Start your agent (uv or docker; see above)

# Run tests against your running agent URL
uv run pytest --agent-url http://localhost:9009
```

## Publishing

The repository includes a GitHub Actions workflow that automatically builds, tests, and publishes a Docker image of your agent to GitHub Container Registry.

If your agent needs API keys or other secrets, add them in Settings → Secrets and variables → Actions → Repository secrets. They'll be available as environment variables during CI tests.

- **Push to `main`** → publishes `latest` tag:
```
ghcr.io/<your-username>/<your-repo-name>:latest
```

- **Create a git tag** (e.g. `git tag v1.0.0 && git push origin v1.0.0`) → publishes version tags:
```
ghcr.io/<your-username>/<your-repo-name>:1.0.0
ghcr.io/<your-username>/<your-repo-name>:1
```

Once the workflow completes, find your Docker image in the Packages section (right sidebar of your repository). Configure the package visibility in package settings.

> **Note:** Organization repositories may need package write permissions enabled manually (Settings → Actions → General). Version tags must follow [semantic versioning](https://semver.org/) (e.g., `v1.0.0`).
