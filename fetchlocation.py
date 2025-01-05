import spacy
import os
import sys

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy language model not found. Please run 'python -m spacy download en_core_web_sm'")
    sys.exit(1)

dataset_folder = "dataset"
directions_folder = os.path.join(dataset_folder, "directions")
objects_folder = os.path.join(dataset_folder, "objects")

opposite_directions = {
    "north": "south", "south": "north",
    "east": "west", "west": "east",
    "northeast": "southwest", "southwest": "northeast",
    "northwest": "southeast", "southeast": "northwest",
    "up": "down", "down": "up",
    "left": "right", "right": "left"
}

def analyze_sentence(sentence):
    doc = nlp(sentence)
    image_locations = []
    directions = set(opposite_directions.keys())
    
    verb_without_direction = False
    expecting_opposite_direction = False

    for token in doc:
        word = token.text.lower()
        
        if word in directions:
            if expecting_opposite_direction:
                opposite_direction = opposite_directions.get(word)
                path = os.path.join(directions_folder, f"{opposite_direction}.png")
                if os.path.exists(path):
                    image_locations.append(f"{opposite_direction} - {path}")
                expecting_opposite_direction = False
            else:
                path = os.path.join(directions_folder, f"{word}.png")
                if os.path.exists(path):
                    image_locations.append(f"{word} - {path}")
            verb_without_direction = False  
        elif token.pos_ == "VERB" and token.dep_ != "aux":
            verb_without_direction = True
            path = os.path.join(objects_folder, f"{word}.png")
            if os.path.exists(path):
                image_locations.append(f"{word} - {path}")
        elif word == "from":
            expecting_opposite_direction = True
            verb_without_direction = False
        else:
            if verb_without_direction:
                path = os.path.join(directions_folder, "right.png")
                if os.path.exists(path):
                    image_locations.append(f"right - {path}")
                verb_without_direction = False  

            path = os.path.join(objects_folder, f"{word}.png")
            if os.path.exists(path):
                image_locations.append(f"{word} - {path}")
    
    if verb_without_direction:
        path = os.path.join(directions_folder, "right.png")
        if os.path.exists(path):
            image_locations.append(f"right - {path}")
    
    return image_locations

def main():
    if len(sys.argv) != 2:
        print("Please provide a sentence as a command-line argument.")
        sys.exit(1)

    sentence = sys.argv[1]
    
    image_locations = analyze_sentence(sentence)
    
    for location in image_locations:
        print(location)

if __name__ == "__main__":
    main()