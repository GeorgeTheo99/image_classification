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
                'General': ['Landscape photography', 'Wildlife photography', 'Astrophotography', 'Storm photography', 'Macro photography', 'Underwater photography'],
                'Landscape photography': ['Mountain Landscapes', 'Coastal Landscapes', 'Urban Landscapes'],
                'Wildlife photography': ['Animal Photography', 'Plant Photography', 'Insect Photography'],
            },
            'Cityscape photography': {
                'General': ['Street photography', 'Structural photography', 'Night cityscape photography'],
                'Architecture photography': ['Modern Architecture', 'Historical Architecture', 'Interior Architecture'],
            },
            'Photographs of people': {
                'General': ['Street photography', 'Portrait photography', 'Fashion photography', 'Event photography', 'Family photography', 'Documentary photography'],
                'Portrait photography': ['Studio Portraits', 'Outdoor Portraits', 'Candid Portraits'],
                'Event photography': ['Wedding photography, Corporate event photography, Concert photography', 'Sports event photography']
            },
            'Artistic photography': {
                'General': ['Still life photography', 'Fine art photography', 'Double exposure photography', 'Surreal photography', 'Abstract photography', 'Macro photography'],
            },
            'Drone/aerial photography': {
            },
            'Food photography': {
                'General': ['Dessert photography', 'Food photography', 'Beverage Photography']
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
    
    def classify_all(self, im, labels):
        image = self.preprocess(im).unsqueeze(0).to(self.device)
        text = clip.tokenize(labels).to(self.device)

        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        class_dict = {}
        print("Classification probabilities:")
        for i, label in enumerate(labels):
            print(f"{label} = {probs[0, i] * 100:.2f}%")
            class_dict[label] = probs[0, i] * 100
        print()

        return class_dict
    
    def classify_sorted(self, im, labels):
        image = self.preprocess(im).unsqueeze(0).to(self.device)
        text = clip.tokenize(labels).to(self.device)

        with torch.no_grad():
            logits_per_image, _ = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy().flatten()

        sorted_probs = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
        print(sorted_probs)
        print()
        return sorted_probs

    def get_photo_quality(self, photo_path):
        im = Image.open(photo_path)
        labels = ["great picture", "average picture", "subpar picture"]
        quality_dict = self.classify_all(im, labels)
        return quality_dict
  
    def get_photo_category(self, photo_path):
        im = Image.open(photo_path)
        category_dict = {}

        # Classify level 1
        level1_sorted = self.classify_sorted(im, list(self.category_map.keys()))
        level1_categories = [label for label, prob in level1_sorted if prob >= level1_sorted[0][1] - 0.05]

        category_dict['level_1'] = set(level1_categories)

        # Process each Level 1 category
        for level1 in level1_categories:
            # Level 2 - if available
            level2_labels = self.category_map[level1].get('General', [])
            if level2_labels:
                level2_sorted = self.classify_sorted(im, level2_labels)
                level2_categories = [label for label, prob in level2_sorted if prob >= level2_sorted[0][1] - 0.05]
                category_dict.setdefault('level_2', set()).update(level2_categories)

                # Process each Level 2 category
                for level2 in level2_categories:
                    # Level 3 classification (if available)
                    if level2 in self.category_map[level1]:
                        level3_labels = self.category_map[level1][level2]
                        if level3_labels:
                            level3_sorted = self.classify_sorted(im, level3_labels)
                            level3_categories = [label for label, prob in level3_sorted if prob >= level3_sorted[0][1] - 0.05]
                            category_dict.setdefault('level_3', set()).update(level3_categories)

        return category_dict

if __name__ == '__main__':
    im = Image.open("/home/georgetheodosopoulos/CVII_project/user_images/Theodosopoulos_George_Security.jpg")
    x = ImageClassifier()
    x.classify_all(im, ['drone/aerial photography', 'photographs of people'])