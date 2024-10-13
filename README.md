# Meal Planner App 🍽️

## Description

The Meal Planner App is a humorous and intelligent assistant that helps users plan their meals, manage their grocery inventory, and provide personalized meal recommendations. It uses advanced AI technology to create witty, customized meal plans while considering user preferences, dietary restrictions, and available ingredients.

## Features

- 🧠 AI-powered meal plan generation
- 📸 Grocery receipt scanning and inventory management
- 🔍 Customizable user preferences
- 😂 Humorous interactions and witty responses
- 💾 Save and load favorite meal plans
- 📅 Daily cooking follow-ups
- ❓ Follow-up questions and plan customization

## Technologies Used

- Streamlit for the user interface
- Google's Generative AI (Gemini) for natural language processing
- ChromaDB for vector database management
- OpenCV and Tesseract for receipt scanning and text extraction
- JSON for data storage

## Setup and Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your Google API in a new `config.py` file:
   ```python
   GOOGLE_API_KEY = "your_api_key_here"
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```

To test the OCR input the given test image.



