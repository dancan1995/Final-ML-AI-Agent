# AI Therapist

## Table of Contents

1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Features](#features)
4. [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup Instructions](#setup-instructions)
5. [Usage](#usage)
    - [How to Run](#how-to-run)
    - [Interacting with the AI Therapist](#interacting-with-the-ai-therapist)
6. [Technologies Used](#technologies-used)
7. [Contributing](#contributing)
8. [Contact](#contact)

---

## Introduction

AI Therapist is a cutting-edge chatbot designed to assist individuals with mental health support, offering conversations focused on conditions like **depression**, **ADHD**, and general emotional well-being. The bot provides personalized responses based on psychological assessments like the **PHQ-9** and leverages AI techniques to understand and engage users effectively.

---

## Project Overview

This project is an AI-powered mental health chatbot designed to interact with users, assess their emotional state, and provide support for various mental health conditions. The chatbot uses data-driven models to simulate therapeutic conversations and offer feedback based on psychological questionnaires, such as the PHQ-9 (Patient Health Questionnaire-9), which is commonly used for diagnosing depression.

The primary goal of this project is to offer a tool that can be used for initial mental health support, helping users self-assess their symptoms and encouraging them to seek professional help if needed.

---

## Features

- **PHQ-9 Assessment**: The bot uses the PHQ-9 questionnaire to assess the user's emotional and mental state.
- **Personalized Responses**: Based on the PHQ-9 results, the chatbot provides personalized advice and responses.
- **Mental Health Support**: Offers feedback on emotional well-being and coping strategies for conditions like depression and ADHD.
- **Data-driven Insights**: Uses AI and machine learning to track user responses and provide recommendations.
- **Interactive Interface**: Engaging text-based conversations with a friendly and approachable personality.

---

## Installation

### Prerequisites

Before you start, make sure you have the following installed:

- **Python** (version 3.7 or higher)
- **pip** (Python package manager)
- **Git** (if you plan to clone this repository)

Additionally, you will need the following Python libraries:

- `flask` (for web app framework)
- `nltk` (Natural Language Toolkit)
- `tensorflow` or `transformers` (for AI/ML models)
- `pandas` (for data manipulation)
- `scikit-learn` (for machine learning models)
- `openai` (if using OpenAI models for conversational responses)

You can install these libraries using the following command:

```bash
pip install -r requirements.txt
```
## Setup Instructions

### Clone the repository:
Clone the repository to your local machine using Git:
```bash
`git clone https://github.com/dancan1995/AI_Therapist.git`
```
### Navigate to the project directory:
```bash
cd AI_Therapist
```

### Install dependencies:
If you haven’t already installed the dependencies, run:

```bash
pip install -r requirements.txt
```

---

## Usage

### How to Run
To start the chatbot, run the following command:

```bash
python main.py
```

This will launch the chatbot in your terminal or web browser, depending on how the application is set up (for example, if you're using a Flask-based web application).

### Interacting with the AI Therapist
Once the chatbot is running, you can interact with it by answering questions related to your mental health status. The bot will ask you to answer a series of questions based on the PHQ-9 assessment, which includes queries about:

- Interest in activities
- Energy levels
- Sleep patterns
- Feelings of hopelessness

Based on your responses, the chatbot will evaluate the severity of your condition (e.g., depression) and provide feedback, support, or guidance.

---

## Technologies Used

- **Python**: The main programming language for the project.
- **Flask**: A micro web framework used for creating the web app (if applicable).
- **TensorFlow/PyTorch**: Machine learning frameworks for building AI models (if applicable).
- **Natural Language Processing (NLP)**: Using libraries like NLTK and transformers for understanding and generating text.
- **OpenAI GPT**: If integrated, for creating conversational agents.
- **scikit-learn**: For machine learning algorithms and preprocessing.
- **Pandas**: For handling data and PHQ-9 responses.

---

## Contributing

We welcome contributions! If you'd like to contribute to the project, follow these steps:

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:

```bash
git clone https://github.com/your-username/AI_Therapist.git
```

3. **Create a new branch** for your feature or fix:

```bash
git checkout -b feature-name
```

4. **Make your changes** and commit them:

```bash
git commit -m "Added a new feature"
```

5. **Push your changes** to your fork:

```bash
git push origin feature-name
```

6. **Create a pull request** from your fork to the main repository.


## Contact

- Dancun Juma (jumad@mail.gvsu.edu) [LinkedIn](https://www.linkedin.com/in/dancun-juma-366403102/)
- Brenda Ondieki (ondiekib@mail.gvsu.edu)
