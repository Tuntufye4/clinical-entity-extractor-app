from django.shortcuts import render
from .forms import ClinicalNoteForm
import spacy
import pandas as pd

# Load Med7 and English models
med7 = spacy.load("en_core_med7_lg")
general_nlp = spacy.load("en_core_web_sm")

# Define colors for labels
col_dict = {}
seven_colours = ['#e6194B', '#3cb44b', '#ffe119', '#ffd8b1', '#f58231', '#f032e6', '#42d4f4']
for label, colour in zip(med7.pipe_labels['ner'], seven_colours):
    col_dict[label] = colour
col_dict["PERSON"] = "#800000"  # Add color for PERSON entity

options = {'ents': med7.pipe_labels['ner'] + ["PERSON"], 'colors': col_dict}

def extract_clinical_entities(text):
    """Extract and visualize clinical entities from the text."""
    # Process text using both models
    med7_doc = med7(text)
    general_doc = general_nlp(text)

    # Merge entities from both models
    entities = list(med7_doc.ents)
    for ent in general_doc.ents:
        if ent.label_ == "PERSON":
            entities.append(ent)

    # Validate merged entities and assign them to the Med7 document
    valid_entities = []
    for ent in entities:
        span = med7_doc.char_span(ent.start_char, ent.end_char, label=ent.label_)
        if span is not None:
            valid_entities.append(span)

    med7_doc.ents = valid_entities

    # Extract entities into a dictionary for further use
    extracted_data = {label: [] for label in options['ents']}
    for ent in med7_doc.ents:
        extracted_data[ent.label_].append(ent.text)

    # Save visualization as an HTML file (optional, for debugging or offline review)
    html = spacy.displacy.render(med7_doc, style="ent", options=options)
    output_path = "ner_visualization_combined.html"
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html)

    return extracted_data

def clinical_extraction_view(request):
    """Handle clinical notes input and extraction."""
    extracted_data = None
    table_html = None

    if request.method == "POST":
        form = ClinicalNoteForm(request.POST)
        if form.is_valid():
            clinical_note = form.cleaned_data["clinical_note"]
            extracted_data = extract_clinical_entities(clinical_note)

            # Remove None values from the lists in the extracted_data
            for key in extracted_data:
                extracted_data[key] = [value for value in extracted_data[key] if value is not None]
                # Convert lists to comma-separated strings
                extracted_data[key] = ", ".join(extracted_data[key])

            # Convert the extracted data into a DataFrame
            df = pd.DataFrame([extracted_data])

            # Convert the DataFrame to an HTML table
            table_html = df.to_html(index=False)
            
    else:
        form = ClinicalNoteForm()

    return render(
        request,
        "extractor/extract.html",
        {"form": form, "extracted_data": extracted_data, "table_html": table_html if extracted_data else None},
    )
