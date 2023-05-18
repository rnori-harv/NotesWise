import PyPDF2
from dotenv import dotenv_values
import openai

env_vars = dotenv_values('.env')

# Access the values
openai.api_key = env_vars['OPENAI_API_KEY']


def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        ans = ""

        for page_number in range(num_pages):
            page = reader.pages[page_number]
            text = page.extract_text()
            ans += text
        return ans

def pass_to_openai(text):
    prompt = "This is your knowledge base, only use the following content in this prompt for your answers. If a question response cannot be found within the text provided, respond to the question with \"NOT POSSIBLE TO ANSWER \". Here is your knowledge base: " + text + "Question: What is a Y-combinator?"
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        n=1,
        stop=None
    )
    generated_text = response.choices[0].text.strip()
    return generated_text


# Provide the path to your PDF file
pdf_file_path = './152/lec01-intro.pdf'
pased_text = read_pdf(pdf_file_path)
print(pass_to_openai(pased_text))
