import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Hello from langchain-course!")
    print(os.environ.get("OPENAI_API_KEY"))

    info = """

        
"Conscientes de que la derrota era inevitable, los Forerunner decidieron destruir todo. Irónicamente, lo hicieron para preservar la vida"...
~Cortana hablando sobre el fin de la Guerra Forerunner-Flood

La Guerra Flood fue un conflicto entre los Forerunner y el Flood. Inició en Seaward en el 97,764 A.N.E. [1] y duró aproximadamente 300 años. [2]

El conflicto culminó con la activación de los Halos, lo cual acabó con la amenaza Flood y los mismos Forerunner, además de todas las formas de vida inteligente, a excepción de algunos especímenes protegidos. [3]

    """


    summary_template = """

    Dada la información {information} acerca de un evento historico quiero que:

    1. Un resumen corto.
    2. 2 datos interesantes acerca del evento.

    """

    summary_prompt_template = PromptTemplate(
        input_variables="information",
        template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    chain = summary_prompt_template | llm
    response = chain.invoke(input= {"information": info})

    print(response.content)
    



if __name__ == "__main__":
    main()
