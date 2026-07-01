from rag import add_pdf_to_vectorstore, search_documents

# Step 1: Add blood report PDF to ChromaDB
result = add_pdf_to_vectorstore(r"C:\Users\acer\Downloads\small_blood_report_sample.pdf")
print(result)

# Step 2: Search it
result = search_documents.invoke("what are the blood test results")
print(result)