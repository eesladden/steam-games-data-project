# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Sets LLM model and temperature
llm = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite", temperature = 0.1)

# Title and introduction
st.title("Steam Games Data Analysis")
st.write("I'll first be asking an initial question: What attributes of a Steam game are associated with a positive user rating (>80% positive reviews), and can they be used to predict it?")