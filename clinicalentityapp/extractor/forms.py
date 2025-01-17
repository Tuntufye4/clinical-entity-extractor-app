from django import forms

class ClinicalNoteForm(forms.Form):
    clinical_note = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 10, "cols": 50}),
        label="Enter Clinical Notes",
    )

