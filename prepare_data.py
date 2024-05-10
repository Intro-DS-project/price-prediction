import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch

import location

def encode_street(street):
    street = location.standardize_street_name(street)
    if street not in location.get_all_streets():
        return -1
    street_encoded = location.get_all_streets().index(street)
    
    return street_encoded

def encode_ward(ward):
    ward = location.standardize_ward_name(ward)
    if ward not in location.get_all_wards():
        return -1
    ward_encoded = location.get_all_wards().index(ward)
    
    return ward_encoded

def encode_district(district):
    district = location.standardize_district_name(district)
    if district not in location.get_all_districts():
        return -1
    district_encoded = location.get_all_districts().index(district)
    
    return district_encoded

def clean_data(supabase_data):
    datas = []
    for data in supabase_data:
        if encode_street(data["street"]) == -1:
            continue
        if encode_ward(data["ward"]) == -1:
            continue
        if encode_district(data["district"]) == -1:
            data["district"] = location.get_district_from_ward(data["ward"])
            if data["district"]:
                print(data["district"])
                datas.append(data)
                continue
        datas.append(data)
    return datas

class RentDataset(Dataset):
    def __init__(self, supabase_response, area_mean=None, area_std=None):
       self.data = supabase_response
       self.area_mean = area_mean
       self.area_std = area_std
       
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        price = torch.Tensor([self.data[idx]["price"]])
        if self.area_mean:
            area = (self.data[idx]["area"] - self.area_mean) / self.area_std
        else:
            area = self.data[idx]["area"]
        
        street = encode_street(self.data[idx]["street"])
        ward = encode_ward(self.data[idx]["ward"])
        district = encode_district(self.data[idx]["district"])
        
        num_bedroom = self.data[idx]["num_bedroom"]
        num_diningroom = self.data[idx]["num_diningroom"]
        num_kitchen = self.data[idx]["num_kitchen"]
        num_toilet = self.data[idx]["num_toilet"]
        
        attr = torch.Tensor([area, num_bedroom, num_diningroom, num_kitchen, num_toilet])
        
        return attr, street, ward, district, price


