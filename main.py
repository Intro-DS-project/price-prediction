import os
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

import torch
from torch import nn
import pickle

import prepare_data
from location import get_district_from_ward

class RentModel(nn.Module):
    def __init__(self, embedding_dims=128, out_feature2=729, out_feature3=81, activation="relu"):
        super().__init__()
        self.embedding_street = nn.Embedding(1572, embedding_dims)   # 1572 streets
        self.embedding_ward = nn.Embedding(430, embedding_dims)   # 430 wards
        self.embedding_district = nn.Embedding(25, embedding_dims)   # 25 districts
        
        self.linear_attr = nn.Linear(5, embedding_dims)
        
        self.linear2 = nn.Linear(embedding_dims*4, out_feature2)
        self.linear3 = nn.Linear(out_feature2, out_feature3)
        self.linear4 = nn.Linear(out_feature3, 1)
        
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, attr, street, ward, district):
        street_embeded = self.embedding_street(street)
        ward_embeded = self.embedding_ward(ward)
        district_embeded = self.embedding_district(district)
        attr_embeded = self.linear_attr(attr)
        
        x = torch.cat((street_embeded, ward_embeded, district_embeded, attr_embeded), 1)
        x = self.dropout(self.act(self.linear2(x)))
        x = self.act(self.linear3(x))
        x = self.linear4(x)
        
        return x
    

class Item(BaseModel):
    area: int
    num_bedroom: int
    num_diningroom: int
    num_kitchen: int
    num_toilet: int
    street: str
    ward: str
    district: str

app = FastAPI()

checkpoint = {}

@app.on_event("startup")
async def load_model():
    print("start")
    with open(os.path.join("model", "area.pkl"), "rb") as f:
        dictionary = pickle.load(f)
    checkpoint["area_mean"], checkpoint["area_std"] = dictionary["area_mean"], dictionary["area_std"]

    with open(os.path.join("model", "ckpt_path.pkl"), "rb") as f:
        dictionary = pickle.load(f)
    ckpt_path = dictionary["ckpt_path"]

    try:
        ckpt = torch.load(ckpt_path)
    except Exception as e:
        print(e)
        ckpt = torch.load("model/checkpoint.pt")

    current_config = ckpt["config"]
    checkpoint["model"] = RentModel(current_config["embedding_dims"], current_config["out_feature2"], current_config["out_feature3"], current_config["activation"])
    checkpoint["model"].load_state_dict(ckpt['model_state_dict'])
    checkpoint["model"].eval()


@app.on_event("shutdown")
async def clear_model():
    print("shutdown")
    checkpoint.clear()


@app.post("/predict")
async def predict(item: Item):
    # Validate
    if item.area < 0 or item.num_bedroom < 0 or item.num_diningroom < 0 or item.num_kitchen < 0 or item.num_toilet < 0:
        return {"success": False, "message": "Values can't be lower than 0!"}
    # Encode street
    encoded_street = prepare_data.encode_street(item.street)
    if encoded_street == -1:
        return {"success": False, "message": "Unrecognized street name!"}
    
    # Encode ward
    encoded_ward = prepare_data.encode_ward(item.ward)
    if encoded_ward == -1:
        return {"success": False, "message": "Unrecognized ward name!"}
    
    # Encode district
    encoded_district = prepare_data.encode_district(item.district)
    if encoded_district == -1:
        district = get_district_from_ward(item.ward)
        if district:
            encoded_district = prepare_data.encode_district(district)
        else:
            return {"success": False, "message": "Unrecognized district name!"}
        
    
    # Normalize area
    area = (item.area - checkpoint["area_mean"]) / checkpoint["area_std"]

    # wrap data
    attr = torch.Tensor([area, item.num_bedroom, item.num_diningroom, item.num_kitchen, item.num_toilet]).unsqueeze(0)
    street = torch.LongTensor([encoded_street])
    ward = torch.LongTensor([encoded_ward])
    district = torch.LongTensor([encoded_district])

    # Predict
    with torch.inference_mode():
        price = checkpoint["model"](attr, street, ward, district)

    # Return 
    return {"success": True, "price": price.item()}
