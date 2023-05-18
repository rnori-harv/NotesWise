import PyPDF2

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)

        for page_number in range(num_pages):
            page = reader.pages[page_number]
            text = page.extract_text()
            print(f"Page {page_number + 1}:")
            print(text)
            print()

# Provide the path to your PDF file
pdf_file_path = './152/lec01-intro.pdf'
read_pdf(pdf_file_path)
