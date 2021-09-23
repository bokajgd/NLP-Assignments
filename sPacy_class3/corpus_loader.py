from pathlib import Path

def corpus_loader(folder: str, full_corpus: bool = False):
    """
    A corpus loader function which takes in a path to a 
    folder and returns a list of strings.
    """
    all_texts = []
    # Load files from directory
    for filename in Path(folder).glob("*.txt"):
        with open(filename, "r", encoding="utf-8") as file:
            text = file.read()
            all_texts.append(text)
            
    # Create corpus from all texts [IF WE WANT ALL TEXTS TOGETHER]
    if full_corpus == True:
        corpus = " ".join(all_texts) 
          
        return corpus 
    
    else: 

        return all_texts