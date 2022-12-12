import re
import csv
from glob import glob # liste tous les fichiers d'un dossier à partir d'un pattern
import os 
import os.path


def nettoyage(file):
    with open(f"{file}", encoding='UTF-8') as txt:
        text = txt.read()
        with open(f"{file}_clean.txt", "w", encoding='UTF-8') as output:
            str_clean = re.sub(r"\b\d+\t+", "", text)
            # str_clean = re.sub(r'\n\n', '\n', str_clean)
            output.write(str_clean)

def transformation_csv(file, language: str):
    """txt to csv

    Args:
        file : fichier clean pour chaque langue
        language (str): le nom de la langue
    """
    # Faire attention à chaque fois que l'on relance : supprimer le fichier data.csv (append)
    
    file_exists = os.path.isfile('data.csv')
    
    with open(file, 'r', encoding='UTF-8') as txt, open('data.csv', 'a', encoding='UTF-8') as csvfile:
        fieldnames = ['Texte', 'Langue'] 
        writer = csv.DictWriter(csvfile, delimiter=';', fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  
        for sent in txt.read().splitlines():
            writer.writerow({'Texte': sent, 'Langue': language})

#["Icelandic", "Danish", "Deutsch", "English", "Estonian", "Finnish", "French", "Croatian", "Hungarian", "Italian", "Latvian", "Dutch", "Polish", "Portuguese", "Romanian", "Slovak", "Slovenian"]

if __name__ == "__main__":
    for file in glob("data/*.txt"):
        nettoyage(file)
        transformation_csv(f'{file}_clean.txt',os.path.basename(file).split(".")[0])