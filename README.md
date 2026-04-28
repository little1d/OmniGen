# SpectrumAnnotator
SupportInfo-Crawl data is available under `crawl_data`.

A flexible data generation framework for creating high-quality machine learning training data. SpectrumAnnotator supports multiple data types including benchmarks, image-text pairs, question-answer pairs, and chain-of-thought reasoning data.

## Quick Start

### 1. Installation

```bash
git clone <https://github.com/little1d/SpectrumAnnotator>
cd SpectrumAnnotator
pip install -e .
```

### 2. Configuration

SpectrumAnnotator supports multiple data generation types. Here are some example configurations:

#### Image-Text Pair Generation

```yaml
# src/config/image_pair.yaml
dataset_path: SpectrumWorld/molpuzzle-seed-datasets
generator_type: image_pair
generator_config:
  prompt_templates:
    basic_description:
      template: |
        You are analyzing a molecular spectrum image. Generate a descriptive English text.
        IMPORTANT CONSTRAINTS - YOU MUST USE THESE EXACT VALUES:
        - Molecular formula: {formula} (DO NOT CHANGE THIS)
        - SMILES notation: {smiles} (DO NOT CHANGE THIS)
        - Spectrum type: {spectrum_type}
model_type: internvl
model_config:
  api_key: "${INTERNVL_API_KEY}"
  base_url: "${INTERNVL_BASE_URL}"
  model_name: "${INTERNVL_MODEL_NAME}"
```

#### QA Pair Generation

```yaml
# src/config/qa_pair.yaml
dataset_path: SpectrumWorld/molpuzzle-seed-datasets
generator_type: qa_pair
generator_config:
  prompt_templates:
    single_step:
      template: |
        You are analyzing a molecular spectrum image. Generate a diverse, natural single-step question-answer conversation about this spectrum.
        TASK: Generate a natural conversation in the following JSON format:
        [{"from": "human", "value": "Your question here"}, {"from": "gpt", "value": "Your response here"}]
model_type: internvl
model_config:
  api_key: "${INTERNVL_API_KEY}"
  base_url: "${INTERNVL_BASE_URL}"
  model_name: "${INTERNVL_MODEL_NAME}"
```

#### Chain-of-Thought Generation

```yaml
# src/config/cot.yaml
dataset_path: SpectrumWorld/molpuzzle-seed-datasets
generator_type: cot
generator_config:
  stages: ["draft", "reason"]
  draft:
    model_type: internvl
    max_out_len: 2000
  reason:
    model_type: interns1
    max_out_len: 3000
model_type: internvl
model_config:
  api_key: "${INTERNVL_API_KEY}"
  base_url: "${INTERNVL_BASE_URL}"
  model_name: "${INTERNVL_MODEL_NAME}"
```

### 3. Environment Setup

Create a `.env` file with your API credentials:

```env
# InternVL Configuration
INTERNVL_API_KEY=your_api_key_here
INTERNVL_BASE_URL=https://api.example.com
INTERNVL_MODEL_NAME=internvl2.5

# InternS1 Configuration (for CoT reasoning)
INTERNS1_API_KEY=your_api_key_here
INTERNS1_BASE_URL=https://api.example.com
INTERNS1_MODEL_NAME=intern-s1

# Gemini Configuration (optional)
GEMINI_API_KEY=your_api_key_here
GEMINI_BASE_URL=https://api.example.com
GEMINI_MODEL_NAME=gemini-2.5-pro
```

### 4. Generate Data

```bash
# Generate image-text pairs
python -m main.generate --type image_pair --samples 10

# Generate QA pairs (single-step)
python -m main.generate --type qa_pair --qa-type single_step --samples 5

# Generate QA pairs (multi-step)
python -m main.generate --type qa_pair --qa-type multi_step --samples 5

# Generate CoT data (text-only, draft stage)
python -m main.generate --type cot --cot-type text_only --stages draft --samples 3

# Generate CoT data (multimodal, both stages)
python -m main.generate --type cot --cot-type multimodal --stages draft,reason --samples 3

# Generate benchmark data
python -m main.generate --type benchmark --samples 10
```

## Output Formats

### Image-Text Pairs

```json
{
  "id": "sample_001",
  "type": "image_text_pair",
  "image": "path/to/spectrum.png",
  "text": "This is a H-NMR spectrum of a molecule with formula C4H8O2 (SMILES: CCOC(=O)C). The spectrum shows three distinct peaks..."
}
```

### QA Pairs

```json
{
  "id": "sample_001",
  "type": "QA pair/single step",
  "image": "path/to/spectrum.png",
  "conversations": [
    {
      "from": "human",
      "value": "What can you tell me about this H-NMR spectrum?"
    },
    {
      "from": "gpt",
      "value": "Based on the spectral features, this appears to be C4H8O2 (SMILES: CCOC(=O)C)..."
    }
  ]
}
```

### Chain-of-Thought Data

```json
{
  "id": "sample_001",
  "type": "cot text-only",
  "question": "What is the molecular structure of this compound?",
  "solution": "The compound is C4H8O2 (SMILES: CCOC(=O)C)...",
  "interns1_thinking_trajectories": "Let me analyze this step by step...",
  "interns1_attempt": "Based on my reasoning, the answer is..."
}
```

### Benchmark Data

```json
[
  {
    "question": "What type of spectrum is this?",
    "choices": [
      "Infrared Spectrum (IR)",
      "Proton Nuclear Magnetic Resonance (H-NMR)",
      "Mass Spectrometry (MS)",
      "Carbon-13 Nuclear Magnetic Resonance (C-NMR)"
    ],
    "answer": "Proton Nuclear Magnetic Resonance (H-NMR)",
    "images": {
      "H-NMR": "output/temp_images/sample_0_h_nmr.png"
    }
  }
]
```

## Data Validation

Validate generated data to ensure quality:

```bash
# Validate image-text pairs
python -m main.validate --type image_pair --file output/image_pairs.json

# Validate QA pairs
python -m main.validate --type qa_pair --file output/qa_pairs.json

# Validate CoT data
python -m main.validate --type cot --file output/cot_data.json

# Validate benchmark data
python -m main.validate --type benchmark --file output/benchmark.json
```

## Supported Models

- **InternVL**: For image-text pair and QA pair generation
- **InternS1**: For chain-of-thought reasoning
- **Gemini**: Alternative model for various tasks

## Features

- **Multiple Data Types**: Support for benchmarks, image-text pairs, QA pairs, and CoT data
- **Flexible Configuration**: YAML-based configuration system
- **Quality Validation**: Built-in validation system for data quality assessment
- **Multi-Model Support**: Integration with InternVL, InternS1, and Gemini
- **Modular Design**: Easy to extend with new generators and validators
