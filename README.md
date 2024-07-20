# Sentiment Scoring Model

## Project Overview

This project focuses on performing sentiment analysis on Amazon reviews using natural language processing (NLP) techniques. It includes various steps, from data exploration and preprocessing to building and evaluating sentiment models.

## Problem Statement

Sentiment analysis of Amazon reviews presents challenges due to the varying lengths and styles of reviews, as well as the need to accurately capture nuanced sentiments expressed in text. The project aims to develop robust models that can effectively analyze sentiment and provide valuable insights from customer reviews.

## Notebooks

### 1. Dataset Creation (Notebook 1/6)
This notebook contains:
- Basic data exploration.
- Class rebalancing.

### 2. Data Preprocessing (Notebook 2/6)
This notebook covers:
- Data clean-up, including removing HTML elements, punctuation, and handling case.
- Data preprocessing techniques such as tokenization, stemming, and lemmatization.

### 3. Sentimental Baseline Model (Notebook 3/6)
This notebook includes:
- Creation of a baseline model that scores reviews based on the frequency of positive and negative words.
- A function that looks up synsets in WordNet, retrieves corresponding SentiWordNet synsets, and calculates sentiment scores.
- Summation and averaging of sentiment scores to obtain the final sentiment score of the input text.
- Visualizations to check the distribution of scores across the dataset.
- Correlation tests between ratings and sentiment scores using various methods.
- Creation of sentiment labels based on threshold values.
- Ground truth labels based on overall ratings.
- Assessment of performance using a confusion matrix.
- Positive and negative sentiment prediction assessment.

### 4. Opinion Lexicon Score Model (Notebook 5/6)
This notebook covers:
- Creation of a sentiment scoring model using an opinion lexicon, a collection of words annotated with sentiment information.
- Breakdown of input text into sentences and calculation of sentence scores based on the presence of positive or negative words from the lexicon.
- Summation of sentence scores to obtain the total sentiment score.

### 5. Analysis on Opinion Lexicon Scores (Notebook 6/6)
This notebook includes:
- Visualization of score distributions across the dataset.
- Correlation tests between ratings and sentiment scores using various methods.
- Creation of sentiment labels based on threshold values.
- Ground truth labels based on overall ratings.
- Assessment of performance using a confusion matrix.
- Positive and negative sentiment prediction assessment.

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/nlp-sentiment-analysis-on-amazon-reviews.git
   ```

2. **Navigate to the project directory:**
    ```bash
    cd nlp-sentiment-analysis-on-amazon-reviews
    ```

3. **Run the notebooks in the following order:**
- `Notebook 1 - Dataset Creation.ipynb`
- `Notebook 2 - Data Preprocessing.ipynb`
- `Notebook 3 - Sentimental Baseline Model.ipynb`
- `Notebook 4 - Opinion Lexicon Score Model.ipynb`
- `Notebook 5 - Analysis on Opinion Lexicon Scores.ipynb`

4. **Results:**
- The baseline model uses WordNet and SentiWordNet to score sentiment based on word frequencies and their sentiment scores.
- The opinion lexicon model scores sentiment based on the presence of positive and negative words in a predefined lexicon.
- Both models' performances are assessed using various visualization techniques, correlation tests, and confusion matrices.

5. **Contributing:**
Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## Conclusion
This project demonstrates effective techniques for sentiment analysis on Amazon reviews using NLP methodologies. The developed models provide insights into customer sentiments, aiding in understanding product reception and identifying areas for improvement. Contributions and further enhancements are encouraged to refine and expand the capabilities of sentiment analysis in real-world applications.
