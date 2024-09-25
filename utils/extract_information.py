import os
import PyPDF2
from llama_parse import LlamaParse

import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv()

high_level_prompt = """You are extracting high-level information from the first page of a scientific article about insomnia. Your task is to output the title and a high-level summary of the article based on the relevant information available on the first page. Follow these guidelines:

1. Title: Extract the title as it appears on the first page.
2. Summary: Provide a concise, high-level summary of the article's content based on the abstract or any other available information from the first page.

Format the output in markdown:

# Title: the title of the Article

## Summary: The high-level summary using the abstract or first-page content.

Important Note:
- The title is very important and must always be included.
- Do not include the ```markdown``` code block syntax."""

content_prompt = """Your task is to extract and convert information from a scientific article about insomnia into markdown format. You must following these guidelines strictly:
Content Guidelines:
- Remove irrelevant information like author details, affiliations, footnotes, citations, URLs, and references.
- Retain tables in markdown format, ensuring accuracy in their structure and content.
- Preserve the original text exactly as presented - no paraphrasing or rewording.
Markdown Formatting:
- Use clear, structured markdown based on the article's formatting. 
- Only create headers when explicitly present (e.g., titles, section headers, sub-headers). Do not create headers that are not in the original text.
- Do not include the ```markdown``` code block syntax.
- Exclude any citations or references (e.g., [1], (Smith et al., 2020), (2020)). Simply skip them without altering the surrounding content.
- Do not introduce additional punctuation (e.g., periods) unless present in the original text.
Handling Page Transitions: The article consists of multiple pages, and you are processing only one page at a time. You must:
- Assume page continuation: Treat the beginning of each page as a continuation of the previous page (not create new headers, not capitalize,...) unless there are clear indicators of a new paragraph or section (e.g., capitalization, new section header, or formatting change).
- End of page considerations: Assume that the end of each page is not the end of a paragraph or section unless there is final punctuation (e.g., a period) or another clear marker (e.g., the end of a sentence or paragraph). Do not add period unless it is present."""



def extract_from_pdf(pdf_path, content_prompt, high_level_prompt, save_dir='./extracted'):
    #Get the first page from PDF file (extract high level information from this page such as title, abstract)
    assert pdf_path.endswith('.pdf'), "Requires PDF file as input"
    with open(pdf_path, 'rb') as input_pdf:
        pdf_reader = PyPDF2.PdfReader(input_pdf)
        output_pdf = PyPDF2.PdfWriter()
        output_pdf.add_page(pdf_reader.pages[0])
        with open('temp.pdf', 'wb') as output_file:
            output_pdf.write(output_file)

    filename = os.path.join(pdf_path.split('/')[-2], pdf_path.split('/')[-1][:-4])
    filename = os.path.join(save_dir, filename)
    #parser for extract content from PDF file
    content_parser = LlamaParse(
        result_type="markdown",
        parsing_instruction=content_prompt
    )
    #parser for extract high level information from PDF file
    high_level_parser = LlamaParse(
        result_type="markdown",
        parsing_instruction=high_level_prompt
    )

    high_level_content = high_level_parser.load_data('temp.pdf')[0].text
    with open(f'{filename}_high_level.md', 'w') as f:
        f.write(high_level_content)

    documents = content_parser.load_data(pdf_path)
    result = ''

    for i, doc in enumerate(documents):
        result += doc.text + ' '
    
    with open(f'{filename}.md', 'w') as f:
        f.write(result)
