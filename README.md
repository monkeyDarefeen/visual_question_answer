# visual_question_answer

## Overview

This project provides a solution for generating questions based on any given image. The questions are crafted to reflect both the overall context of the image and the individual activities within it. The approach focuses on **data augmentation** rather than fine-tuning models, offering enhanced generalizability across various image scenarios.

### Key Features

1. **Data Augmentation**: Expands the dataset to cover diverse scenarios and contexts.
2. **Context Extraction**: Utilizes advanced models to understand the full context of the image.
3. **Question Generation and Answer Analysis**: Creates meaningful questions from the image and evaluates generated answers.

### Models Used

- **LLaVA v1.6 Mistral 7B (llava-v1.6-mistral-7b-hf)**: For image-language understanding.
- **Segment Anything Model (sam_vit_h_4b8939)**: For segmenting images into meaningful parts.
- **Meta LLaMA 3.1-405B Instruct (meta/llama-3.1-405b-instruct)**: For advanced question generation.
  > **Note:** You need to provide your own API key for `meta/llama-3.1-405b-instruct`. The API key must be added in the `utils\load_llama.py` file in the `api_key=""` field.

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

### Steps

1. Clone the repository:

   ```bash
   git clone <repository-link>
   cd <repository-directory>
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up the **Meta LLaMA 3.1 API key**:
   - Open the `utils\load_llama` file.
   - Locate the line `api_key=""` and set it to your API key:
     ```python
     api_key="your_api_key_here"
     ```

---

## How the Notebook Works

1. **Open the Notebook**:  
   Open `main.ipynb` in Jupyter Notebook, JupyterLab, or any IDE supporting `.ipynb` files.

2. **Input an Image**:  
   Place the desired image in the input directory as specified in the notebook.

3. **Run the Cells**:  
   Execute the cells in order. The notebook performs the following steps:

   - **Data Augmentation**: Prepares augmented datasets for better image analysis.
   - **Context Extraction**: Extracts meaningful insights from the image using the specified models.
   - **Question Generation**: Generates a set of questions reflecting the image's context and individual activities.
   - **Answer Analysis**: Evaluates and analyzes the generated answers.

4. **Output**:
   - Augmented image datasets.
   - Extracted contextual information.
   - A list of generated questions and their corresponding answers.

---

## Contributing

Contributions to enhance the project are welcome! Follow these steps:

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/YourFeature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Your message here"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/YourFeature
   ```
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

Let me know if further changes are needed!
