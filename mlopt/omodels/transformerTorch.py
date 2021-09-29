import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import math
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)


def generate_square_subsequent_mask(sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

class TransformerDecoderModel(pl.LightningModule):
    '''
    Autoregresive Decoder-Only Transformer. Variant number 1.
    '''
    def __init__(self, input_size, output_size, n_features, d_model=256, nhead=8, num_layers=3, dropout=0.1):
        super(TransformerDecoderModel, self).__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.criterion = nn.L1Loss()
        self.warmup_steps = 4000

        self.output_size = output_size
        self.n_features = n_features

        self.encoder = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.decoder = nn.Linear(n_features, d_model)
        self.pos_decoder = PositionalEncoding(d_model, dropout) 

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,dim_feedforward=d_model*4, dropout=dropout, activation='relu')
        self.transformer_decoder  = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, n_features)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None
        
    def forward(self, src,trg):

        src = src.permute(0,1)
        trg = trg.permute(0,1)

        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = generate_square_subsequent_mask(len(trg)).to(trg.device)

        src = self.encoder(src)
        src = self.pos_encoder(src)

        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        output = self.transformer_decoder(trg, src, tgt_mask=self.trg_mask, memory_mask=self.src_mask)
        output = self.fc_out(output)

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.shape[0], x.shape[1], 1)
        y = torch.cat((x, y),1)
        
        y_hat = self(x,y)
        loss = self.criterion(y_hat, y[:])
        self.log("train_loss",loss,on_epoch=True,logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        decoderInput = x[:, -1].unsqueeze(-1)
        for i in range(0,self.output_size):
            out = self(x,decoderInput)
            decoderInput = torch.cat((decoderInput,out[:,-1].unsqueeze(-1).detach()),1) 
            
        y_hat = decoderInput[:, 1:]
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss,on_epoch=True,prog_bar=True,logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9)
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
        on_tpu=False, using_native_amp=False, using_lbfgs=False):

        d_model, steps, warmup_steps = self.d_model, self.global_step + 1, self.warmup_steps
        rate = (d_model ** (- 0.5)) * min(steps ** (- 0.5), steps * warmup_steps ** (- 1.5))

        for pg in optimizer.param_groups:
            pg['lr'] = rate

        optimizer.step(closure=optimizer_closure)


class TransformerDecoderModel2(pl.LightningModule):
    '''
    Autoregresive Decoder-Only Transformer. Variant number 2.
    '''
    def __init__(self,input_size,output_size, n_features, d_model=256,nhead=8, num_layers=3, dropout=0.1):
        super(TransformerDecoderModel2, self).__init__()

        self.d_model = d_model
        self.criterion = nn.L1Loss()
        self.warmup_steps = 4000

        self.output_size = output_size
        self.n_features = n_features

        self.encoder = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,dim_feedforward=d_model*4, dropout=dropout, activation='relu')
        self.transformer_decoder  = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, n_features)

        self.src_mask = None
        
    def forward(self, src):

        src = src.permute(1,0)

        if self.src_mask is None or self.src_mask.size(0) != len(src):
            self.src_mask = generate_square_subsequent_mask(len(src)).to(src.device)

        src = self.encoder(src)
        src = self.pos_encoder(src)

        output = self.transformer_decoder(src,self.src_mask)
        output = self.fc_out(output)

        return output.permute(1,0)

    def training_step(self, batch, batch_idx):
        x,y = batch
        z = torch.cat((x,y[:,:-1]),1)
        
        y_hat = self(z)
        y_hat = y_hat[:,x.shape[1]:]
        loss = self.criterion(y_hat, y[:, 1:])
        self.log("train_loss",loss,on_epoch=True,logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = x
        for i in range(0,self.output_size):
            out = self(z)
            z = torch.cat((z,out[:,-1].unsqueeze(-1).detach()),1) 
            
        y_hat = z[:,x.shape[1]:]
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss,on_epoch=True,prog_bar=True,logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9)
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
        on_tpu=False, using_native_amp=False, using_lbfgs=False):

        d_model, steps, warmup_steps = self.d_model, self.global_step + 1, self.warmup_steps
        rate = (d_model ** (- 0.5)) * min(steps ** (- 0.5), steps * warmup_steps ** (- 1.5))

        for pg in optimizer.param_groups:
            pg['lr'] = rate

        optimizer.step(closure=optimizer_closure)

class transformerEncoderDecoder(pl.LightningModule):
    '''
    Full Transformer
    '''
    def __init__(self,input_size, output_size, n_features, d_model=256, nhead=8, num_layers=3, dropout=0.1):
        super(transformerEncoderDecoder, self).__init__()

        self.d_model = d_model
        self.criterion = nn.L1Loss()
        self.warmup_steps = 4000
        
        self.encoder = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.decoder = nn.Linear(input_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,dim_feedforward=d_model*4 , dropout=dropout, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,dim_feedforward=d_model*4, dropout=dropout, activation='relu')
        self.transformer_decoder  = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(n_features, output_size)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None
        
    def forward(self, src, trg):

        src = src.permute(1,0)
        trg = trg.permute(1,0)

        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = generate_square_subsequent_mask(len(trg)).to(trg.device)

        src = self.encoder(src)
        src = self.pos_encoder(src)

        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        memory = self.transformer_encoder(src,self.src_mask)

        output = self.transformer_decoder(trg,memory, tgt_mask=self.trg_mask, memory_mask=self.src_mask)
        output = self.fc_out(output)

        return output.permute(1,0)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.cat((x[:,-1,:].unsqueeze(1),y ),1)
        
        y_hat = self(x,y[:, :-1])
        loss = self.criterion(y_hat, y[:, 1:])
        self.log("train_loss",loss,on_epoch=True,logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        decoderInput = x[:, -1].unsqueeze(-1)
        for i in range(0,self.output_size):
            out = self(x,decoderInput)
            decoderInput = torch.cat((decoderInput,out[:,-1].unsqueeze(-1).detach()),1) 
            
        y_hat = decoderInput[:, 1:]
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss,on_epoch=True,prog_bar=True,logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9)
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
        on_tpu=False, using_native_amp=False, using_lbfgs=False):

        d_model, steps, warmup_steps = self.d_model, self.global_step + 1, self.warmup_steps
        rate = (d_model ** (- 0.5)) * min(steps ** (- 0.5), steps * warmup_steps ** (- 1.5))

        for pg in optimizer.param_groups:
            pg['lr'] = rate

        optimizer.step(closure=optimizer_closure)

class TransformerEncoderModel(pl.LightningModule):
    '''
    Non-Autoregresive encoder Transformer + MLP head
    '''
    def __init__(self,input_size, output_size, n_features, d_model=256, nhead=8, num_layers=3, dropout=0.1):
        super(TransformerEncoderModel, self).__init__()

        self.d_model = d_model
        self.criterion = nn.L1Loss()
        self.warmup_steps = 4000

        self.output_size = output_size
        self.n_features = n_features

        self.encoder = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,dim_feedforward=d_model*4, dropout=dropout, activation='relu')
        self.transformer_decoder  = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model*input_size, output_size*n_features)

        self.src_mask = None
        
    def forward(self, src):

        src = self.encoder(src)
        src = self.pos_encoder(src)

        src = src.permute(1,0)
        output = self.transformer_decoder(src)
        output = output.permute(1,0)

        output = torch.flatten(output,1)
        output = self.fc_out(output)

        return output.view(-1,self.output_size,self.n_features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss",loss,on_epoch=True,logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss,on_epoch=True,prog_bar=True,logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9)
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
        on_tpu=False, using_native_amp=False, using_lbfgs=False):

        d_model, steps, warmup_steps = self.d_model, self.global_step + 1, self.warmup_steps
        rate = (d_model ** (- 0.5)) * min(steps ** (- 0.5), steps * warmup_steps ** (- 1.5))

        for pg in optimizer.param_groups:
            pg['lr'] = rate

        optimizer.step(closure=optimizer_closure)

class TransformerTorch():

    def __init__(self, N=2, d_model=24, h=8) -> None:
        self._model_factory = {
        "trD_AR": self.TransformerDecoder,
        "trD2_AR": self.TransformerDecoder2,
        "trE": self.TransformerEncoder
        }
        self._N = N
        self._d_model = d_model
        self._h = h
        pass
    
    def TransformerDecoder(self,input_shape,output_size):
        model = TransformerDecoderModel(input_shape[-2],output_size,input_shape[-1],self._d_model,self._h,self._N)
        return model

    def TransformerDecoder2(self,input_shape,output_size):
        model = TransformerDecoderModel2(input_shape[-2],output_size,input_shape[-1],self._d_model,self._h,self._N)
        return model

    def TransformerEncoder(self,input_shape,output_size):
        model = TransformerEncoderModel(input_shape[-2],output_size,input_shape[-1],self._d_model,self._h,self._N)
        return model

    
    def create_model(self, model_name, input_shape, batch_size=256, max_steps_per_epoch=256, **args):
        assert model_name in self._model_factory.keys(), "Model '{}' not supported".format(
            model_name
        )
        return self._model_factory[model_name](input_shape, **args)

    def trainAndTest(self, x_train, y_train, x_test, y_test,
                     model_name="trD_AR", batch_size=256, max_steps_per_epoch=256,
                     epochs=400, gpu_device=0):
        
        forecast_horizon = y_test.shape[1]

        steps_per_epoch = min(
            int(np.ceil(x_train.shape[0] / batch_size)), max_steps_per_epoch,
        )

        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).float().unsqueeze(-1)

        x_test = torch.from_numpy(x_test).float()
        y_test = torch.from_numpy(y_test).float().unsqueeze(-1)

        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_test, y_test)

        train_loader = DataLoader(train_dataset, 
                                    batch_size = batch_size, 
                                    shuffle = False)

        val_loader = DataLoader(val_dataset, 
                                    batch_size = batch_size, 
                                    shuffle = False)

        test_loader = DataLoader(val_dataset, 
                                    batch_size = batch_size, 
                                    shuffle = False)

        trainer = Trainer(max_epochs=epochs,max_steps=steps_per_epoch, gpus=[gpu_device],checkpoint_callback=False)

        print("Creating model")


        model = self.create_model(model_name,x_train.shape,output_size=forecast_horizon,
                                  batch_size=batch_size,max_steps_per_epoch=max_steps_per_epoch)

        trainer.fit(model,train_loader,val_loader)
        print("End training")
        train_loss = float(trainer.callback_metrics["train_loss"].to("cpu"))
        val_loss = float(trainer.callback_metrics["val_loss"].to("cpu"))
        
        print("train_loss: ")
        print(train_loss)
        print("val_loss: ")
        print(val_loss)
        print("loss saved")