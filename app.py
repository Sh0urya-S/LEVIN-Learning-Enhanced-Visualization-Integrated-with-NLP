import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import subprocess
import os
import spacy
import threading
import sounddevice as sd
import soundfile as sf
import queue
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio

# Load pre-trained Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

PYTHON_SCRIPT_PATH = r"C:\Users\sshou\Desktop\finalyearproject\pyfiles\levinproj\fetchlocation.py"
DIRECTIONS_FOLDER = os.path.join("dataset", "directions")
OBJECTS_FOLDER = os.path.join("dataset", "objects")

nlp = spacy.load("en_core_web_sm")

# Configuration
filename = "recorded_audio.wav"
samplerate = 44100  # Sample rate for recording
q = queue.Queue()

# Global variables
recording = False
audio_data = None

class ImageLayer:
    def __init__(self, name, image_path, canvas, layer_panel, x_position, y_position, resize_large=False):
        self.name = name
        self.image_path = image_path
        self.canvas = canvas
        self.layer_panel = layer_panel
        self.x_position = x_position
        self.y_position = y_position

        image_width = 1000 if resize_large else 100
        self.image = Image.open(image_path).resize((image_width, 100))
        self.tk_image = ImageTk.PhotoImage(self.image)

        self.canvas_id = self.canvas.create_image(self.x_position, self.y_position, image=self.tk_image)
        self.label_id = self.canvas.create_text(self.x_position, self.y_position + 60, text=self.name, fill="black")

        self.layer_frame = tk.Frame(layer_panel, bd=2, relief="solid", padx=10, pady=10)
        self.layer_frame.pack(pady=5, fill=tk.X, padx=5)

        tk.Label(self.layer_frame, text=self.name, font=("Helvetica", 14)).pack(side=tk.LEFT)
        close_button = tk.Button(self.layer_frame, text="Close", command=self.close_layer, font=("Helvetica", 12))
        close_button.pack(side=tk.RIGHT)

    def close_layer(self):
        self.canvas.delete(self.canvas_id)
        self.canvas.delete(self.label_id)
        self.layer_frame.destroy()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Levin Alpha")
        self.root.geometry("1600x1000")  
        self.root.configure(bg="#FFDAB9")
        
        self.title_label = tk.Label(self.root, text="Levin Alpha Build v.1.0.21", font=("Helvetica", 20, "bold"), bg="#FFDAB9")
        self.title_label.grid(row=0, column=0, columnspan=3, pady=10, sticky="n")  

        # Create a scrollable canvas
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.grid(row=1, column=0, rowspan=2, padx=10, pady=10)

        self.canvas_scrollbar_y = tk.Scrollbar(self.canvas_frame, orient="vertical")
        self.canvas_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas_scrollbar_x = tk.Scrollbar(self.canvas_frame, orient="horizontal")
        self.canvas_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas = tk.Canvas(self.canvas_frame, width=1000, height=700, bg="white", yscrollcommand=self.canvas_scrollbar_y.set, xscrollcommand=self.canvas_scrollbar_x.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas_scrollbar_y.config(command=self.canvas.yview)
        self.canvas_scrollbar_x.config(command=self.canvas.xview)
         
        self.canvas = tk.Canvas(root, width=1000, height=700, bg="white")
        self.canvas.grid(row=1, column=0, rowspan=2, padx=10, pady=10)

        self.button_frame = tk.Frame(self.root, bg="#FFDAB9")
        self.button_frame.grid(row=3, column=0, columnspan=2, pady=20)  

        self.record_button = tk.Button(
            self.button_frame,
            text="Record",
            command=self.start_recording,
            width=20,  
            height=3,  
            font=("Helvetica", 16)  
        )
        self.record_button.pack(side=tk.LEFT, padx=20)  

        self.stop_button = tk.Button(
            self.button_frame,
            text="Stop",
            command=self.stop_recording,
            width=20,
            height=3,
            font=("Helvetica", 16)
        )
        self.stop_button.pack(side=tk.LEFT, padx=20)
        self.stop_button.config(state=tk.DISABLED)

        self.analyze_button = tk.Button(
            self.button_frame,
            text="Generate",
            command=self.analyze_recording,
            width=20,
            height=3,
            font=("Helvetica", 16)
        )
        self.analyze_button.pack(side=tk.LEFT, padx=20)
        self.analyze_button.config(state=tk.DISABLED)

        self.layer_panel_container = tk.Frame(root, width=250, height=700, bg="#FFDAB9")
        self.layer_panel_container.grid(row=1, column=2, rowspan=2, padx=10, pady=10, sticky="n")
        self.layer_panel_container.grid_propagate(False)

        self.layer_panel = tk.Frame(self.layer_panel_container, bg="#FFDAB9")
        self.layer_panel.pack(fill=tk.BOTH, expand=True)

        self.layers_label = tk.Label(self.layer_panel, text="Layers:", font=("Helvetica", 16, "bold"), bg="#FFDAB9")
        self.layers_label.pack(pady=0)

        self.loaded_images = []
        self.notification_textbox = tk.Text(self.root, height=8, width=50,font=("Helvetica", 14))
        self.notification_textbox.grid(row=3, column=2, padx=10, pady=10)

    def start_recording(self):
        global recording
        if recording:
            self.notification_textbox.insert(tk.END, "Recording is already in progress.\n")
            return

        self.record_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        recording = True
        threading.Thread(target=self.record_audio).start()

    def record_audio(self):
        global recording
        self.notification_textbox.insert(tk.END, "Recording...\n")
        with sf.SoundFile(filename, mode='w', samplerate=samplerate, channels=1) as file:
            with sd.InputStream(samplerate=samplerate, channels=1, callback=self.audio_callback):
                while recording:
                    file.write(q.get())
        self.notification_textbox.insert(tk.END, "Recording complete.\n")

    def stop_recording(self):
        global recording
        self.record_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        recording = False

        if os.path.exists(filename):
            self.notification_textbox.insert(tk.END, f"Audio saved to {filename}\n")
        else:
            self.notification_textbox.insert(tk.END, "No audio file found.\n")

        self.analyze_button.config(state=tk.NORMAL)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        q.put(indata.copy())

    def analyze_recording(self):
        if not os.path.exists(filename):
            self.notification_textbox.insert(tk.END, "No recorded audio file found.\n")
            return

        try:
            transcription = self.evaluate(filename)
            transcription = transcription.lower()
            self.notification_textbox.insert(tk.END, f"Transcribed text: {transcription}\n")
            self.generate_images(transcription)

        except Exception as e:
            self.notification_textbox.insert(tk.END, f"Error during transcription: {e}\n")

    def evaluate(self, audio_path):
        model.eval()
        waveform, sr = torchaudio.load(audio_path)
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform).squeeze(0)

        # Preprocess the audio
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

        # Perform inference
        with torch.no_grad():
            logits = model(inputs.input_values).logits

        # Decode the predicted IDs to text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        return transcription

    def determine_positions(self, sentence):
        doc = nlp(sentence)
        positions = {}
        current_x1 = 150
        current_x2 = 1000-150
        static_x = 500
        current_y1 = 150
        current_y2 = 700-150
        static_y = 400
        below_y = static_y + 150
        is_below = False
        has_direction = False
        large_objects = set()
        direction_map = {
            "north": "north",
            "south": "south",
            "east": "east",
            "west": "west",
            "right": "right",
            "left": "left",
            "up": "up",
            "down": "down",
            "northeast": "northeast",
            "northwest": "northwest",
            "southeast": "southeast",
            "southwest": "southwest"
        }
        arrows = {}
        for i, token in enumerate(doc):
            word = token.text.lower()
            if token.pos_ == "NOUN":
                if i + 2 < len(doc) and doc[i + 2].text.lower() in ["left", "west"]:
                    positions[word] = (current_x2, static_y)
                    current_x2 -= 120
                elif i + 2 < len(doc) and doc[i + 2].text.lower() in ["up", "north"]:
                    positions[word] = (static_x, current_y2)
                    current_y2 -= 120
                elif i + 2 < len(doc) and doc[i + 2].text.lower() in ["down", "south"]:
                    positions[word] = (static_x, current_y1)
                    current_y1 += 120
                elif is_below:
                    positions[word] = (500, below_y)
                    below_y += 120
                    is_below = False
                    large_objects.add(word)
                else:
                    positions[word] = (current_x1, static_y)
                    current_x1 += 120
            elif token.pos_ == "ADP":
                if word in ["on"]:
                    is_below = True
            elif word in direction_map:
                if word in ["left", "west"]:
                    positions[word] = (current_x2, static_y)
                    arrows[word] = word
                    current_x2 -= 120
                    has_direction = True
                elif word in ["up", "north"]:
                    positions[word] = (static_x, current_y2)
                    arrows[word] = word
                    current_y2 -= 120
                    has_direction = True
                elif word in ["down", "south"]:
                    positions[word] = (static_x, current_y1)
                    arrows[word] = word
                    current_y1 += 120
                    has_direction = True
                else:
                    positions[word] = (current_x1, static_y)
                    arrows[word] = word
                    current_x1 += 120
                    has_direction = True

        if not has_direction:
            positions["right"] = (current_x1, static_y)
            current_x1 += 120

        return positions, large_objects, arrows

    def generate_images(self, input_text):
        if not input_text:
            messagebox.showwarning("Input Error", "Please enter a sentence!")
            return

        try:
            process = subprocess.Popen(
                ["python", PYTHON_SCRIPT_PATH, input_text],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()

            if stderr:
                messagebox.showerror("Python Error", stderr)
                return

            self.canvas.delete("all")
            self.clear_layers()

            image_positions, large_objects, arrows = self.determine_positions(input_text)

            lines = stdout.splitlines()
            for line in lines:
                if " - " in line:
                    name, path = line.split(" - ")
                    if os.path.exists(path):
                        resize_large = name in large_objects
                        x, y = image_positions.get(name, (150, 100))
                        self.add_image_layer(name, path, x, y, resize_large)

                        if name in arrows:
                            direction = arrows[name]
                            arrow_image_path = os.path.join(DIRECTIONS_FOLDER, f"{direction}_arrow.png")
                            if os.path.exists(arrow_image_path):
                                self.add_image_layer(f"{direction}_arrow", arrow_image_path, x, y + 120)

            self.canvas.update()
            self.layer_panel.update()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during image generation: {e}")

    def clear_layers(self):
        for layer in self.loaded_images:
            layer.layer_frame.destroy()
        self.loaded_images.clear()

    def add_image_layer(self, name, image_path, x_position, y_position, resize_large=False):
        layer = ImageLayer(name, image_path, self.canvas, self.layer_panel, x_position, y_position, resize_large)
        self.loaded_images.append(layer)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()