import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import re


def sort_by_lower_bound(age_range):
    lower_bound = int(re.search(r'\d+', age_range).group())
    return lower_bound


class FairfaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(root_dir + '/' + csv_file)
        self.root_dir = root_dir
        self.transform = transform

        self.label_age_mapping = {label: idx for idx, label in
                                  enumerate(sorted(self.data_frame['age'].unique(), key=sort_by_lower_bound))}
        self.label_gender_mapping = {label: idx for idx, label in
                                     enumerate(self.data_frame['gender'].unique())}
        self.label_race_mapping = {label: idx for idx, label in
                                   enumerate(self.data_frame['race'].unique())}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx, 0]
        img_path = f"{self.root_dir}/{img_name}"
        image = Image.open(img_path)
        age = self.label_age_mapping[self.data_frame.iloc[idx, 1]]
        gender = self.label_gender_mapping[self.data_frame.iloc[idx, 2]]
        race = self.label_race_mapping[self.data_frame.iloc[idx, 3]]
        if self.transform:
            image = self.transform(image)
        return image, age, gender, race

    def __getdist__(self):
        age_dist = self.data_frame['age'].value_counts()
        gender_dist = self.data_frame['gender'].value_counts()
        race_dist = self.data_frame['race'].value_counts()
        return age_dist, gender_dist, race_dist


if __name__ == "__main__":
    # df = pd.read_csv('datasets/Fairface/fairface_label_train.csv')
    # unique_labels = df['age'].unique()
    # sorted_list = sorted(unique_labels, key=sort_by_lower_bound)
    # dict = {label: idx for idx, label in enumerate(sorted_list)}
    # print(dict)
    pass
