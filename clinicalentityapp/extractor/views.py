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

# Store cumulative extracted data
cumulative_data = []


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

    # Filter and resolve overlapping entities
    entities = sorted(entities, key=lambda x: (x.start_char, -x.end_char))  # Sort by start, longest span first
    resolved_entities = []
    last_end = -1

    for ent in entities:
        if ent.start_char >= last_end:  # Avoid overlaps
            resolved_entities.append(ent)
            last_end = ent.end_char

    # Assign resolved entities to med7_doc
    valid_entities = []
    for ent in resolved_entities:
        span = med7_doc.char_span(ent.start_char, ent.end_char, label=ent.label_)
        if span is not None:
            valid_entities.append(span)

    med7_doc.ents = valid_entities

    # Extract entities into a dictionary for further use
    extracted_data = {label: [] for label in options['ents']}
    for ent in med7_doc.ents:
        extracted_data[ent.label_].append(ent.text)

    return extracted_data


def clinical_extraction_view(request):
    """Handle clinical notes input and extraction."""
    global cumulative_data
    table_html = None

    if request.method == "POST":
        form = ClinicalNoteForm(request.POST)
        if form.is_valid():
            clinical_note = form.cleaned_data["clinical_note"]
            try:
                # Extract entities from the new clinical note
                new_data = extract_clinical_entities(clinical_note)

                # Remove None values from the lists in the extracted_data
                for key in new_data:
                    new_data[key] = [value for value in new_data[key] if value is not None]

                # Check if this data is already in cumulative_data to prevent repetition
                if new_data not in cumulative_data:
                    cumulative_data.append(new_data)

                # Create a DataFrame from cumulative data
                all_data = []
                for data in cumulative_data:
                    flattened = {key: ", ".join(value) for key, value in data.items()}
                    all_data.append(flattened)

                df = pd.DataFrame(all_data)   

                # Move the "PERSON" column to the beginning if it exists
                if "PERSON" in df.columns:
                    columns = ["PERSON"] + [col for col in df.columns if col != "PERSON"]
                    df = df[columns]

                # Convert the DataFrame to an HTML table
                table_html = df.to_html(index=False)

            except Exception as e:
                print(f"Error during entity extraction: {e}")
    else:
        form = ClinicalNoteForm()

    return render(
        request,
        "extractor/extract.html",
        {"form": form, "table_html": table_html},
    )
