import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt

class ImageClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.category_map = {
            'Nature photography': {
                'General': ['Landscape photography', 'Wildlife photography', 'Astrophotography', 'Storm photography', 'Macro photography', 'Flower photography', 'Underwater photography'],
                'Landscape photography': ['Mountain Landscapes', 'Coastal Landscapes', 'Urban Landscapes'],
                'Wildlife photography': ['Bird Photography', 'Mammal Photography', 'Insect Photography'],
            },
            'Cityscape and structural photography': {
                'General': ['Architecture photography', 'Street photography', 'Cityscape photography'],
                'Architecture photography': ['Modern Architecture', 'Historical Architecture'],
            },
            'People oriented photography': {
                'General': ['Street photography', 'Portrait photography', 'Fashion photography', 'Sports photography', 'Event photography', 'Documentary photography'],
                'Portrait photography': ['Studio Portraits', 'Outdoor Portraits', 'Candid Portraits'],
                'Event photography': ['Wedding photography, Corporate event photography']
            },
            'Artistic photography': {
                'General': ['Still life photography', 'Fine art photography', 'Double exposure photography', 'Surreal photography', 'Abstract photography', 'Macro photography'],
            },
            'Drone/aerial photography': {
            },
            'Food photography': {
                'General': ['Dessert photography', 'Food photography']
            }
        }

    def classify_max(self, im, labels):
        image = self.preprocess(im).unsqueeze(0).to(self.device)
        text = clip.tokenize(labels).to(self.device)

        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        print("Classification probabilities:")
        for i, label in enumerate(labels):
            print(f"{label} = {probs[0, i] * 100:.2f}%")
        print()

        max_index = probs.argmax()
        max_label = labels[max_index]
        max_prob = probs[0, max_index]

        print(f"Most likely category: {max_label} = {max_prob * 100:.2f}%")
        return max_label, max_prob

    def get_photo_quality(self, photo_path):
        im = Image.open(photo_path)
        labels = ["great picture", "average picture", "subpar picture"]
        self.classify_max(im, labels)

    def get_photo_category(self, photo_path):
        im = Image.open(photo_path)
        
        category_dict = {str(photo_path): []}

        # Classify level 1
        max_label, _ = self.classify_max(im, list(self.category_map.keys()))
        print(f"Level 1: {max_label}")
        category_dict[str(photo_path)].append({'level_1': {max_label}})

        # Level 2 - if available
        level2_labels = self.category_map[max_label].get('General', [])
        if level2_labels:
            level2_label, _ = self.classify_max(im, level2_labels)
            print(f"Level 2 Category within {max_label}: {level2_label}")
            category_dict[str(photo_path)].append({'level_2': {max_label}})

            # Level 3 classification (if available)
            if level2_label in self.category_map[max_label]:
                level3_labels = self.category_map[max_label].get(level2_label, [])
                if level3_labels:
                    level3_label, _ = self.classify_max(im, level3_labels)
                    print(f"Level 3 Category within {level2_label}: {level3_label}")
                    category_dict[str(photo_path)].append({'level_3': {max_label}})
        
        return category_dict
