import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
from sklearn.model_selection import train_test_split
import ray
from ray import tune

import os
from dotenv import load_dotenv
from supabase import create_client, Client
import datetime
from datetime import date
import pickle

import prepare_data

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

class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss,self).__init__()
        self.eps = eps

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y) + self.eps)
        return loss

def get_dataloader(supabase_data):
    datas = prepare_data.clean_data(supabase_data)
    
    train_data, valid_data = train_test_split(datas, test_size=0.3, random_state=42)
    
    # Normalize area
    list_area = [int(data["area"]) for data in train_data]
    area_mean = np.mean(list_area)
    area_std = np.std(list_area)
    
    train_dataset = prepare_data.RentDataset(train_data, area_mean=area_mean, area_std=area_std)
    valid_dataset = prepare_data.RentDataset(valid_data, area_mean=area_mean, area_std=area_std)
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=True)
    
    return train_dataloader, valid_dataloader

def evaluate(supabase_data, current_area_mean, current_area_std, current_model, threshold=6):
    datas = prepare_data.clean_data(supabase_data)
    dataset = prepare_data.RentDataset(datas, area_mean=current_area_mean, area_std=current_area_std)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    
    loss_fn = RMSELoss()
    
    current_model.eval()
    with torch.inference_mode():
        for batch_idx, (attr, street, ward, district, price) in enumerate(dataloader):
            logits = current_model(attr, street, ward, district)
            loss = loss_fn(logits, price)
    
    if loss.item() > threshold:
        return True, loss.item()   # need to re-train
    return False, loss.item()   # no need to re-train
    

def model_tuning(config, train_dataloader, valid_dataloader, checkpoint_dir=None):
    # train_dataloader, valid_dataloader = get_dataloader(supabase_data)
     
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = RentModel(config["embedding_dims"], config["out_feature2"], config["out_feature3"], config["activation"]).to(device)
    # loss_fn = nn.MSELoss()
    loss_fn = RMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    num_epochs = 50
    print_per_batch = 1
    
    model.train()
    valid_loss_values = []
    for epoch in range(1, num_epochs+1):
        train_loss_values = []
        for batch_idx, (attr, street, ward, district, price) in enumerate(train_dataloader):
            logits = model(attr, street, ward, district)
            loss = loss_fn(logits, price)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % print_per_batch == 0:
                print(
                    f"Epoch: {epoch:03d}/{num_epochs:03d}"
                    f" | Batch {batch_idx:03d}/{len(train_dataloader):03d}"
                    f" | Train Loss: {loss}"
                )
            
            train_loss_values.append(loss.item())
            
        model.eval()
        with torch.inference_mode():
            for batch_idx, (attr, street, ward, district, price) in enumerate(valid_dataloader):
                logits = model(attr, street, ward, district)
                loss = loss_fn(logits, price)
                print(
                        f"Epoch: {epoch:03d}/{num_epochs:03d}"
                        f" | Batch {batch_idx:03d}/{len(valid_dataloader):03d}"
                        f" | Val Loss: {loss}"
                    )
                
                valid_loss_values.append(loss.item())
                
            if len(valid_loss_values) >= 3:
                if valid_loss_values[-1] >= valid_loss_values[-2] and valid_loss_values[-2] >= valid_loss_values[-3]:
                    print("Maybe Overfitting... Stop!")
                    break
        with tune.checkpoint_dir(step=0) as checkpoint_dir:
            model.save(checkpoint_dir)
        tune.report(mean_loss=sum(valid_loss_values) / len(valid_loss_values))
        
if __name__ == '__main__':
    load_dotenv()
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    
    supabase: Client = create_client(url, key)
    
    # Get today's data
    response = supabase.table('entries').select("*").gte("created_at", date.today()).execute()
    
    # Load saved data
    SAVED_FOLDER = "model"
    with open(os.path.join(SAVED_FOLDER, "area.pkl"), "rb") as f:
        dictionary = pickle.load(f)
    area_mean, area_std = dictionary["area_mean"], dictionary["area_std"]
    with open(os.path.join(SAVED_FOLDER, "val_loss.pkl"), "rb") as f:
        dictionary = pickle.load(f)
    val_loss = dictionary["loss"]
    current_model = torch.load(os.path.join(SAVED_FOLDER,"model.pt"))
    
    # Evaluate
    retrain, old_val_loss = evaluate(response.data, area_mean, area_std, current_model, threshold=0.4)
    if retrain:
        print(old_val_loss)
        # Get data within 1 month
        all_response = supabase.table('entries').select("*") \
                                            .gte("created_at", date.today()-datetime.timedelta(days=30)) \
                                            .lt("created_at", date.today()-datetime.timedelta(days=1)) \
                                            .execute()
        
        # train with old data + part of new data. Others for validation                                    
        all_datas = prepare_data.clean_data(all_response.data)
        new_datas = prepare_data.clean_data(response.data)
        
        train_data, valid_data = train_test_split(new_datas, test_size=0.8, random_state=42)
        train_data += all_datas
        
        # Normalize area
        list_area = [int(data["area"]) for data in train_data]
        area_mean = np.mean(list_area)
        area_std = np.std(list_area)
        
        # Prepare dataloader
        train_dataset = prepare_data.RentDataset(train_data, area_mean=area_mean, area_std=area_std)
        valid_dataset = prepare_data.RentDataset(valid_data, area_mean=area_mean, area_std=area_std)
        
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=True)
        
        # Config for tuning
        config = {
            "embedding_dims": tune.grid_search([32, 64, 128]),
            "out_feature2": tune.grid_search([512, 729, 1024]),
            "out_feature3": tune.grid_search([32, 64, 81]),
            "act": tune.choice(["relu", "gelu"]),
            "lr": tune.loguniform(1e-4, 1e-5)
        }
        
        reporter = tune.CLIReporter(metric_columns=["mean_loss", "training_iteration"])
        analysis = tune.run(
            # lambda config: model_tuning(config, train_dataloader, valid_dataloader),
            tune.with_parameters(model_tuning, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader),
            config=config,
            progress_reporter=reporter,
            metric="mean_loss"
        )
        
        print(analysis.best_checkpoint)
        print(analysis.best_result)
        print(analysis.dataframe())
    else:
        print(old_val_loss)
        print("Model is still good")
        
        
    

    
    
    
    
    
