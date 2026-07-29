from django import forms

class UploadCSVForm(forms.Form):
    file = forms.FileField(label='Pilih file CSV')