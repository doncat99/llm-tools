import asyncio
import logging
import argparse

import json

from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import data_gemma as dg

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from prompt_toolkit.styles import Style
from prompt_toolkit import PromptSession

# Initialize Data Commons API client
DC_API_KEY = 'qIQTHmIXlpw4yjQQh7OqokxjYZRKXByaWLPmFKkVnOh576IS'

RAG_IN_CONTEXT_PROMPT = """
Given a QUERY below, your task is to come up with a maximum of 25
STATISTICAL QUESTIONS that help in answering QUERY.

Here are the only forms of STATISTICAL QUESTIONS you can generate:

1. "What is $METRIC in $PLACE?"
2. "What is $METRIC in $PLACE $PLACE_TYPE?"
3. "How has $METRIC changed over time in $PLACE $PLACE_TYPE?"

where:
- $METRIC should a publicly accessible metric on societal topics around
  demographics, economy, health, education, environment, etc.  Examples are
  unemployment rate, life expectancy, etc.
- $PLACE is the name of a place like California, World, Chennai, etc.
- $PLACE_TYPE is an immediate child type within $PLACE, like counties, states,
  districts, etc.

Your response should only include the questions, one per line without any
numbering or bullet!  If you cannot come up with statistical questions to ask,
return an empty response.

NOTE:  Do not repeat questions.  Limit the number of questions to 25.

If QUERY asks about  multiple concepts (e.g., income and diseases), make sure
the questions cover all the concepts.

[Start of Examples]

QUERY: Which grades in the middle school have the lowest enrollment in Palo Alto?
STATISTICAL QUESTIONS:
What is the number of students enrolled in Grade 6 in Palo Alto schools?
What is the number of students enrolled in Grade 7 in Palo Alto schools?
What is the number of students enrolled in Grade 8 in Palo Alto schools?

QUERY: Which industries have grown the most in California?
STATISTICAL QUESTIONS:
How have jobs in agriculture changed over time in California?
How has GDP of agriculture sector changed over time in California?
How have jobs in information and technology changed over time in California?
How has GDP of information and technology sector changed over time in California?
How have jobs in the government changed over time in California?
How has GDP of the government sector changed over time in California?
How have jobs in healthcare changed over time in California?
How has GDP of healthcare sector changed over time in California?
How have jobs in entertainment changed over time in California?
How has GDP of entertainment sector changed over time in California?
How have jobs in retail trade changed over time in California?
How has GDP of retail trade sector changed over time in California?
How have jobs in manufacturing changed over time in California?
How has GDP of manufacturing sector changed over time in California?
How have jobs in education services changed over time in California?
How has GDP of education services sector changed over time in California?

QUERY: Which state in the US has the most asian population?
STATISTICAL QUESTIONS:
What is the number of asian people in US states?

QUERY: Do specific health conditions affect the richer California counties?
STATISTICAL QUESTIONS:
What is the median income among California counties?
What is the median house price among California counties?
What is the prevalence of obesity in California counties?
What is the prevalence of diabetes in California counties?
What is the prevalence of heart disease in California counties?
What is the prevalence of arthritis in California counties?
What is the prevalence of asthma in California counties?
What is the prevalence of chronic kidney disease in California counties?
What is the prevalence of chronic obstructive pulmonary disease in California counties?
What is the prevalence of coronary heart disease in California counties?
What is the prevalence of high blood pressure in California counties?
What is the prevalence of high cholesterol in California counties?
What is the prevalence of stroke in California counties?
What is the prevalence of poor mental health in California counties?
What is the prevalence of poor physical health in California counties?


[End of Examples]

QUERY: {question}
STATISTICAL QUESTIONS:
"""


class DataCommonsClient:
    def __init__(self):
        self.data_fetcher = dg.DataCommons(api_key=DC_API_KEY)

    def call_dc(self, questions: list[str]) -> dict[str, dg.base.DataCommonsCall]:

        try:
            q2resp = self.data_fetcher.calln(questions, self.data_fetcher.point)
        except Exception as e:
            logging.warning(e)
            q2resp = {}
            pass

        return  q2resp
    
    @staticmethod
    def pretty_print(q2resp: dict[str, dg.base.DataCommonsCall]):
        markdown_output = "# Data Commons Response\n"
        for k, v in q2resp.items():
            markdown_output += f"**{k}**\n\n"
            markdown_output += f"{v.answer()}\n\n"
        return markdown_output


class DataGemma:
   def __init__(self, model_id: str = "bartowski/datagemma-rag-27b-it-GGUF", model_file: str =  "datagemma-rag-27b-it-Q2_K.gguf"):
      self.generation_kwargs = {
         "max_tokens": 4096, # Max number of new tokens to generate
      }
      self.model_path = hf_hub_download(model_id, model_file)
      self.llm = Llama(
            self.model_path
      )
      self.name = "DataGemma"
   
   def complete(self, question: str) -> str:
      llm_resp = self.llm(question, **self.generation_kwargs)

      return llm_resp["choices"][0]["text"]
   
   @staticmethod
   def parse_completion(completion: str) -> list[str]:
      return [line for line in completion.split('\n') if line.strip()]
   

class Claude:
   def __init__(self, model_id: str = "claude-3-5-sonnet-20240620"):
      self.llm = ChatAnthropic(model=model_id, max_tokens_to_sample=4000)
      self.standard_parser = StrOutputParser()
      self.name = "Claude Sonnet 3.5"
      
   def complete(self, question: str) -> str:
      prompt_template = PromptTemplate.from_template(RAG_IN_CONTEXT_PROMPT)
      chain =  prompt_template | self.llm | self.standard_parser
      return chain.invoke(question)
   @staticmethod
   def parse_completion(completion: str) -> list[str]:
      return [line for line in completion.split('\n') if line.strip()]


async def get_user_input(prompt="Your question: "):
    style = Style.from_dict({
        'prompt': 'cyan bold',
    })
    session = PromptSession(style=style)
    return await session.prompt_async(prompt, multiline=False)


async def main(llm : DataGemma | Claude, pretty_response: bool, execute_queries: bool):
    console.print(Panel("Welcome to the Retrieval Interleaved Generation (RIG) Demo!", title="Welcome", style="bold green"))

    while True:
        user_input = await get_user_input()
        if user_input.lower() == 'exit':
            console.print(Panel("Thank you for chatting. Goodbye!", title_align="left", title="Goodbye", style="bold green"))
            break
        else:
            #QUERY = "What progress has Pakistan made against health goals?"
            console.print(f"Calling {llm.name} with '{user_input.lower()}'", style="green")
            completion = llm.complete(user_input.lower())
            questions = llm.parse_completion(completion)

            if pretty_response:
                console.print(Panel(
                    Syntax(json.dumps(questions, indent=2), "json", theme="monokai"),
                    title="Data Gemma Response",
                    expand=False
                ))
            else:
                console.print(Panel(
                    Syntax(completion, "text"),
                title="Data Gemma Raw Response",
                expand=False
            ))                

            if execute_queries: 
                dc = DataCommonsClient()

                q2resp = dc.call_dc(questions)
                filtered_q2resp = {}
                for k, v in q2resp.items():
                    if v and v.val != '':
                        filtered_q2resp[k] = v
                    else:
                        console.print(f"Did not find any answer to question '{k}' in Data Commons. Skipping.", style="yellow")
                q2resp = filtered_q2resp

                if not pretty_response:
                    console.print(Panel(
                        Syntax(json.dumps(q2resp, default=lambda o: o.__dict__, indent=2), "json", theme="monokai"),
                    title="Data Commons Raw Response",
                    expand=False
                ))
                else:
                    markdown_output = dc.pretty_print(q2resp)
                    console.print(Panel(
                        Markdown(markdown_output),
                        title="Data Commons Response",
                        expand=False
                    ))


if __name__ == "__main__":
    console = Console()
    
    parser = argparse.ArgumentParser(description="Retrieval Interleaved Generation Demo")
    parser.add_argument("--pretty-response", action="store_true", help="Enable pretty response formatting")
    parser.add_argument("--execute-queries", action="store_true", help="Execute queries against Data Commons")
    parser.add_argument("--model", choices=['data-gemma', 'claude'], default='data-gemma', help="Choose the model to use")
    args = parser.parse_args()
    pretty_response = args.pretty_response
    execute_queries = args.execute_queries
    model = args.model

    if model == 'data-gemma':
        llm = DataGemma()
    elif model == 'claude':
        llm = Claude()

    try:
        asyncio.run(main(llm, pretty_response, execute_queries))
    except KeyboardInterrupt:
        console.print("\nProgram interrupted by user. Exiting...", style="bold red")
    except Exception as e:
        console.print(f"An unexpected error occurred: {str(e)}", style="bold red")
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
    finally:
        console.print("Goodbye!", style="bold green")