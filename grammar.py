import spacy

nlp = spacy.load("en_core_web_sm")

directions = {"north", "south", "east", "west", "northeast", "northwest", 
              "southeast", "southwest", "up", "down", "left", "right"}

def analyze_sentence(sentence):
    doc = nlp(sentence)
    directions_found = []
    objects_found = []
    prepositions = []
    conjunctions = []
    
    for token in doc:
        if token.text.lower() in directions:
            directions_found.append(token.text.lower())
        elif token.pos_ == "NOUN" or token.dep_ in ("nsubj", "dobj", "pobj", "attr"):
            objects_found.append(token.text.lower())
        elif token.pos_ == "CCONJ":
            conjunctions.append(token.text.lower())
        elif token.pos_ == "ADP":
            prepositions.append(token.text.lower())
    
    return directions_found, objects_found, prepositions, conjunctions

if __name__ == "__main__":
    sentence = input("Enter a sentence: ").strip()
    directions_found, objects_found, prepositions, conjunctions = analyze_sentence(sentence)
    
    print("Directions:", directions_found)
    print("Objects:", objects_found)
    print("Prepositions:", prepositions)
    print("Conjunctions:", conjunctions)