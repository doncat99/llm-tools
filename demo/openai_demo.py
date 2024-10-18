

import torch

import textwrap
import data_gemma as dg
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

dc = dg.DataCommons(api_key='qIQTHmIXlpw4yjQQh7OqokxjYZRKXByaWLPmFKkVnOh576IS')
openai_model = dg.OpenAI(model='gpt-4o-mini', api_key='sk-mM2cA3fB7CxjmJbN71E0Bb58F56c451f9c370847E233A6Db', base_url='https://lonlie.plus7.plus/v1')

HF_TOKEN = "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"

# nf4_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_compute_dtype=torch.bfloat16
# )

model_name = 'google/datagemma-rag-27b-it'
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
datagemma_model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                            #  quantization_config=nf4_config,
                                             torch_dtype=torch.bfloat16,
                                             token=HF_TOKEN)

# Build the LLM Model stub to use in RAG flow
datagemma_model_wrapper = dg.HFBasic(datagemma_model, tokenizer)

# @title Pick a query from a sample list{ run: "auto" }
QUERY = "Do the US states with high coal fired power also have high rates of COPD?" #@param ["Do the US states with high coal fired power also have high rates of COPD?", "Is obesity in America continuing to increase?", "Which US states have the highest percentage of uninsured children?", "Which US states have the highest cancer rates?", "How have CO2 emissions changed in France over the last 10 years?", "Which New Jersey schools have the highest student to teacher ratio?", "Show me a breakdown of income distribution for Seattle.", "Which New Jersey cities have the best commute times for workers?", "Does India have more people living in the urban areas or rural areas?  How does that vary by states?  Are the districts with the most urban population also located in the states with the most urban population?", "Can you find a district in India each where: 1. there are more muslims than hindus or christians or sikhs;  2. more christians than the rest;  3. more sikhs than the rest.", "What are some interesting trends in Sunnyvale spanning gender, age, race, immigration, health conditions, economic conditions, crime and education?", "Where are the most violent places in the world?", "Compare Cambridge, MA and Palo Alto, CA in terms of demographics, education, and economy stats.", "Give me some farming statistics about Kern county, CA.", "What is the fraction households below poverty status receive food stamps in the US?  How does that vary across states?", "Is there evidence that single-parent families are more likely to be below the poverty line compared to married-couple families in the US?", "What patterns emerge from statistics on safe birth rates across states in India?", "Are there significant differences in the prevalence of various types of disabilities (such as vision, hearing, mobility, cognitive) between Dallas and Houston?", "Are there states in the US that stand out as outliers in terms of the prevalence of drinking and smoking?", "Has the use of renewables increased globally?", "Has the average lifespan increased globally?"]


def display_chat(prompt, text):
  formatted_prompt = "<font size='+1' color='brown'>🙋‍♂️<blockquote>" + prompt + "</blockquote></font>"
  text = text.replace('•', '  *')
  text = textwrap.indent(text, '> ', predicate=lambda _: True)
  formatted_text = "<font size='+1' color='teal'>🤖\n\n" + text + "\n</font>"
  return f"{formatted_prompt}\n\n{formatted_text}\n\n"

ans = dg.RAGFlow(llm_question=datagemma_model_wrapper, llm_answer=openai_model, data_fetcher=dc).query(query=QUERY)

display_chat(QUERY, ans.answer())