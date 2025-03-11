from torch.utils.data import Dataset
import pandas as pd


class T2IDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            self.lines = [line.strip() for line in file]  # 

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return {"text": self.lines[idx], "index": idx}
    

class Dataset(Dataset):
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path, sep='\t')
        prompts = self.dataset['Prompt'].tolist()
        
        self.lines = [line.strip() for line in prompts]  # 

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return {"text": self.lines[idx], "index": idx}
    

subsampled_data = {
    'promptist': (
    [
     'A rabbit is wearing a space suit.', 
     'Several railroad tracks with one train passing by.',
     'The roof is wet from the rain.',
     'Cats dancing in a space club.'
    ],
    [
     'A rabbit is wearing a space suit, digital Art, Greg rutkowski, Trending cinematographic artstation.',
     'Several railroad tracks with one train passing by, hyperdetailed, artstation, cgsociety, 8k.',
     'The roof is wet from the rain, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration.',
     'Cats dancing in a space club, digital painting, artstation, concept art, soft light, hdri, smooth, sharp focus, illustration, fantasy.'
    ]
    ),
    'beautiful': (
    [
     'Astronaut rides horse.', 
     'A majestic sailing ship.',
     'Sunshine on iced mountain.',
     'Panda mad scientist mixing sparkling chemicals.'
    ],
    [
     'Astronaut riding a horse, fantasy, intricate, elegant, highly detailed, artstation, concept art, smooth, sharp focus, illustration.',
     'A massive sailing ship, by Greg Rutkowski, highly detailed, stunning beautiful photography, unreal engine, 8K.',
     'Photo of sun rays coming from melting iced mountain, by greg rutkowski, 4 k, trending on artstation.',
     'Panda as a mad scientist, lab coat, mixing glowing and disinertchemicals, fantasy, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration.'
    ]
    )
}
# subsampled_data = (
#     ['Astronaut rides horse.', 
#      'A majestic sailing ship.',
#      'Sunshine on iced mountain.',
#      'Panda mad scientist mixing sparkling chemicals.'],
#     ['Astronaut riding a horse, fantasy, intricate, elegant, highly detailed, artstation, concept art, smooth, sharp focus, illustration.',
#      'A massive sailing ship, by Greg Rutkowski, highly detailed, stunning beautiful photography, unreal engine, 8K.',
#      'Photo of sun rays coming from melting iced mountain, by greg rutkowski, 4 k, trending on artstation.',
#      'Panda as a mad scientist, lab coat, mixing glowing and disinertchemicals, fantasy, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration.']
# )