# 📘 QueryGenius - Contextual Navigation Intelligence

## 🚀 Overview

**QueryGenius** is an advanced contextual navigation system powered by AI that predicts and personalizes location categories based on user history, time, and dynamic contextual factors. This intelligent system enhances navigation by understanding patterns and delivering optimized recommendations for places such as schools, offices, restaurants, parks, and gyms.

## 🎯 Key Features

### 🧠 Context-Aware Predictions

- Learns from historical data to predict user location categories.
- Incorporates time-based patterns for better accuracy.

### 📊 Dynamic Adaptation

- Adjusts predictions based on changing user behavior.
- Factors in real-time conditions to refine recommendations.

### 🔍 Smart Categorization

- Classifies locations into categories such as:
  - 🏢 Office
  - 🏫 School
  - 🍽️ Restaurant
  - 🌳 Park
  - 🏋️ Gym

### 🛠️ Data Processing & AI Model

- Utilizes **DeepSeek R1** for reasoning and predictions.
- Implements efficient text chunking and document embedding for better AI understanding.
- Uses **LangChain** for robust document processing.

### 📂 Multi-Source Data Handling

- Supports **PDF documents** for location-based data extraction.
- Enables **web scraping** to enhance location intelligence.

### 💬 Interactive AI Chatbot

- Seamlessly integrates with **Streamlit UI**.
- Engages users with an animated typing effect for responses.
- Dynamically fetches and processes location-related queries.

## 🏗️ System Architecture

### 1️⃣ **Data Collection**

- User history and contextual metadata.
- Uploaded PDFs and scraped websites for additional location insights.

### 2️⃣ **Data Processing**

- **Text Chunking**: RecursiveCharacterTextSplitter for document breakdown.
- **Vector Embeddings**: Converts text into AI-friendly embeddings.
- **Context Matching**: Uses similarity search for relevant location retrieval.

### 3️⃣ **AI Model Integration**

- Utilizes **DeepSeek R1** via **Ollama** for intelligent predictions.
- Generates responses dynamically based on queried context.

### 4️⃣ **User Interaction**

- Streamlit-based chatbot interface for engaging AI-driven responses.
- Supports real-time conversation and navigation assistance.

## 📌 How It Works

1. **Choose Data Source** 📂 🌐

   - Upload a PDF containing location information.
   - Enter a website URL for automated scraping.

2. **Data Processing & Embedding** 🔄

   - Extracts and chunks relevant text.
   - Stores processed information in a vector database.

3. **Query & Response Generation** 💬

   - User submits a navigation-related question.
   - AI retrieves the most relevant data and generates an informed response.

4. **Dynamic Updates** 🔄⚡

   - Learns from user interactions and refines future predictions.

## 🛠️ Technologies Used

- **Python** 🐍 - Core programming language.
- **Streamlit** 🎨 - Interactive UI framework.
- **LangChain** 🔗 - Document processing and chunking.
- **DeepSeek R1 via Ollama** 🤖 - AI-powered reasoning engine.
- **BeautifulSoup** 🌐 - Web scraping utility.
- **InMemoryVectorStore** 🗃️ - Localized vector storage for fast retrieval.

## 🔥 Why Use QueryGenius?

✅ Personalized navigation experience.\
✅ AI-driven predictions improve with usage.\
✅ Supports both **textual documents & live web data**.\
✅ Intuitive chatbot interface for easy access to insights.\
✅ Powered by **cutting-edge AI** for smarter decision-making.

---

