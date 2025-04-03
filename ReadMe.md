# WebAndDocGPT

## Overview
WebAndDocGPT is an AI-powered Streamlit app for contextual data extraction and natural language processing. Users can upload documents or provide web URLs to receive intelligent responses based on the extracted information.

## Features
- **AI-Powered Q&A**: Extracts insights from documents and web content.
- **Multi-Input Support**: Processes both file uploads and URLs.
- **User-Friendly Interface**: Accessible via a Streamlit web UI.

## Supported File Formats
- **Documents**: PDF
- **Web Content**: Extracts text from provided URLs

## Required API Keys
To use WebAndDocGPT, the following API keys are required:
- **OpenAI API Key**: Required for language model processing.
- **SupaBase Key and URL** : For storing and retrieving the embeddings.
- **Google API Key** *(Optional)*: To extract structured data from web pages.

## Installation

Ensure Python 3.8+ is installed, then run:

```sh
git clone https://github.com/KarthikeyanS-4/web_and_doc_GPT.git
cd WebAndDocGPT
pip install -r requirements.txt
```

## Execution

Start the application with:

```sh
streamlit run webanddocGPT.py
```

### Usage
1. Open the Streamlit interface.
2. Upload a document or enter a URL.
3. Ask context-based questions.
4. Receive AI-generated responses.

