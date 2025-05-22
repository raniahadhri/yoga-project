import zipfile

with zipfile.ZipFile('yoga-poses-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('data')

   