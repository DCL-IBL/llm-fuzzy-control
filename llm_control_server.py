from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from peft import LoraConfig, get_peft_model, PeftModel

import pdb
import json

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

global data1

class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    #curl "http://127.0.0.1:8000/test?ID=1&G=5&BW=8&t=0"
    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)  # dict: key -> [values]
        if 'G' in query.keys() and 'BW' in query.keys() and 'ID' in query.keys() and 't' in query.keys():
            global data1
            #pdb.set_trace()
            id = query['ID'][0]
            data1.BW = float(query['BW'][0])
            time = float(query['t'][0])
            if id not in data1.state.keys():
                data1.state[id] = torch.tensor([50.0,50.0,0.0,0.0,0.0,0.0,0.0,0.0]).cuda().repeat(Controller.batch_size,1)
                data1.yt[id] = (data1.state[id][:,0]/(0.16*data1.BW)).unsqueeze(1).repeat(1,Controller.sequence_len*data1.input_toks_per_term)
                data1.ut[id] = torch.zeros((Controller.batch_size,Controller.sequence_len*data1.output_toks_per_term)).cuda()
                data1.t[id] = 0.0
                data1.u_prev[id] = torch.tensor(0.0)
            err =  float(query['G'][0])*(0.16*data1.BW) - data1.state[id][:,0]
            #data1.state[id][:,5] = data1.state[id][:,5]-0.1*err
            data1.state[id][:,0] = torch.tensor(float(query['G'][0])*(0.16*data1.BW))
            u_per_kg = data1.get_control(time,id,0.01*err)
            self._send_json({"u_new_kg": u_per_kg.item()})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b""
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, status=400)
            return

        url = urlparse(self.path)
        query = parse_qs(url.query)
        # minimal example: echo both query and JSON body
        self._send_json({"path": url.path, "query": query, "json": data})

def triangular(x, a, b, c):
    # a < b < c (can be tensors or scalars, broadcastable with x)
    left  = (x - a) / (b - a)
    right = (c - x) / (c - b)
    return torch.clamp(torch.minimum(left, right), min=0.0, max=1.0)

def ramp_right(x, a, b):
    y = (x - a) / (b - a)
    return torch.clamp(y, min=0.0, max=1.0)

def ramp_left(x, a, b):
    y = - (x - b) / (b - a)
    return torch.clamp(y, min=0.0, max=1.0)

def hypo_glucose(x):
    return ramp_left(x,90/18,105/18) # 70 - 105 mg/dl

def target_glucose(x):
    return triangular(x,90/18,105/18,180/18) # 70 - 180 mg/dl

def hyper_glucose(x):
    return ramp_right(x, 150/18, 300/18) # 105 - 180 mg/dl

def zero_dose(x):
    return ramp_left(x,0,0.01) # mU/kg

def low_dose(x):
    return triangular(x, 0.0, 0.01, 0.1) # mU/kg

def high_dose(x):
    return ramp_right(x, 0.05, 0.2) # mU/kg

class Controller():
    llm_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_name = "jkralev/fuzzy-llm"

    input_terms = ['hypoglycemia','in range','hyperglycemia']
    output_terms = ['zero','low','high']
    msg_end_len = 3
    sequence_len = 10
    batch_size = 1
    input_alternatives = batch_size #len(input_terms)*batch_size

    messages1 = [{"role":"system","content":"You are an automatic insulin delivery advisor for type 1 diabetes. "},
           {"role":"user","content":"Historical glucose: "}]
    messages2 = [{"role":"assistant","content":"Next insulin dose: "}]
    messages3 = ". Historical insulin dosages: "

    def __init__(self):
        self.x_vals = torch.linspace(0,0.2,50).cuda().half()
        self.mf_vals = torch.vstack([zero_dose(self.x_vals).unsqueeze(0),low_dose(self.x_vals).unsqueeze(0),high_dose(self.x_vals).unsqueeze(0)])

        self.llm = AutoModelForCausalLM.from_pretrained(Controller.llm_name,device_map="auto")
        #self.llm.load_adapter(Controller.adapter_name)
        #self.llm = self.llm.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(Controller.llm_name)

        self.Nemb = self.llm.config.hidden_size

        tokenized_messages1 = self.tokenizer.apply_chat_template(Controller.messages1, tokenize=True, 
                                                   add_generation_prompt=False,
                                                   continue_final_message=False,
                                                   return_tensors="pt")
        vectorized_messages1 = self.llm.model.embed_tokens(tokenized_messages1['input_ids'][0].cuda()).unsqueeze(0)
        self.vectorized_messages1 = vectorized_messages1.repeat(Controller.input_alternatives,1,1)

        tokenized_messages2 = self.tokenizer.apply_chat_template(Controller.messages2, tokenize=True, 
                                                   add_generation_prompt=False, 
                                                   continue_final_message=True,
                                                   return_tensors="pt")
        vectorized_messages2 = self.llm.model.embed_tokens(tokenized_messages2['input_ids'][0].cuda()).unsqueeze(0)
        self.vectorized_messages2 = vectorized_messages2.repeat(Controller.input_alternatives,1,1)

        tokenized_messages3 = self.tokenizer(Controller.messages3, add_special_tokens=False, return_tensors="pt")
        vectorized_messages3 = self.llm.model.embed_tokens(tokenized_messages3['input_ids'][0].cuda()).unsqueeze(0)
        self.vectorized_messages3 = vectorized_messages3.repeat(Controller.input_alternatives,1,1)

        input_terms_tok = self.tokenizer(Controller.input_terms, add_special_tokens=False, return_tensors="pt", padding=True)
        self.input_toks_per_term = input_terms_tok['input_ids'].shape[1]
        self.input_terms_tok = input_terms_tok['input_ids'].repeat(1,Controller.sequence_len)
        self.input_terms_vec0 = self.llm.model.embed_tokens(self.input_terms_tok.cuda())

        output_terms_tok = self.tokenizer(Controller.output_terms, add_special_tokens=False, return_tensors="pt", padding=True)
        self.output_toks_per_term = output_terms_tok['input_ids'].shape[1]
        self.output_terms_tok = output_terms_tok['input_ids'].repeat(1,Controller.sequence_len)
        self.output_terms_vec0 = self.llm.model.embed_tokens(self.output_terms_tok.cuda())
        self.llm = PeftModel.from_pretrained(self.llm, Controller.adapter_name, is_trainable=True)

        #lora_config = LoraConfig(
        #    r=16,                      # Low rank
        #    lora_alpha=32,             # Scaling factor
        #    target_modules=["q_proj", "v_proj"],  # Layers to apply LoRA
        #    lora_dropout=0.1,
        #    bias="none",
        #    task_type="CAUSAL_LM"
        #)
        #self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()
        
        self.optimizer = torch.optim.AdamW(self.llm.parameters(), lr=1e-7) # 1e-5
        self.llm.train()

        self.BW=100
        self.t={}
        self.u_prev = {}
        self.state = {} #torch.tensor([50,50,0,0,0,0,0,0]).cuda().repeat(Controller.batch_size,1)
        self.yt = {} #(self.state[:,0]/(0.16*self.BW)).unsqueeze(1).repeat(1,Controller.sequence_len*self.input_toks_per_term) #torch.zeros((batch_size,sequence_len)).cuda()
        self.ut = {} #torch.zeros((Controller.batch_size,Controller.sequence_len*self.output_toks_per_term)).cuda()
    
    Nstate = 8
    
    def ap_model(self,state,uin,UG):
        BW = torch.tensor([self.BW]).cuda()
        VG = 0.16*BW # Glucose distribution volume, L/kg*kg = L
        VI = 0.12*BW # Insulin distribution volume, L/kg*kg = L
        F01 = 0.0097*BW # Non-insulin glucose flux, mmol/min/kg*kg = mmol/min
        k12 = 0.0066 # Transfer rate from interstitial to plasma, 1/min
        ka1 = 0.006 # 1/min
        ka2 = 0.06 # 1/min
        ka3 = 0.03 # 1/min
        SIT = 51.2e-4 # 1/min/(mU*1/L) = L/min/mU
        SID = 8.2e-4 # 
        SIE = 520e-4 #
        kb1 = SIT*ka1
        kb2 = SID*ka2
        kb3 = SIE*ka3
        EGP0 = 0.0161*BW # Endogenous glucose production extrapolated to zero insulin concentration, mmol/kg/min*kg = mmol/min

        AG = 0.8 # Carbohydrate bioavailability
        tmaxG = 40 # Time to max carbohydrate, min

        tmaxI = 55 # Time-to-maximum of absorption of subcutaneously injected short-acting insulin, min
        ke = 0.138 # Insulin elimination from plasma, 1/min
    
        Q1 = state[:,0] # glucose masses in plasma, mmol
        Q2 = state[:,1] # glucose masses in interstitial, mmol
        x1 = state[:,2] # Insulin effect on glucose transport and distribution between plasma and interstitial, 1/min
        x2 = state[:,3] # Insulin effect on glucose transport and distribution between plasma and interstitial, 1/min
        x3 = state[:,4] # Insulin effect on glucose transport and distribution between plasma and interstitial, 1/min
        S1 = state[:,5] # Insulin sensitivity in accessible, mU
        S2 = state[:,6] # Insulin sensitivity in nonaccesible, mU
        I = state[:,7] # Insulin in plasma, mU
    
        G = Q1/VG # mmol/L
 
        F01c = F01*torch.ones((Controller.batch_size,)).cuda() # mmol/min
        sel = G>4.5
        if sel.sum().item() > 0:
            F01c[sel] = F01[sel]*G[sel]/4.5 # mmol/min
        FR = torch.zeros((Controller.batch_size,)).cuda()
        sel = G>=9
        if sel.sum().item() > 0:
            FR[sel] = 0.003*(G[sel]-9)*VG[sel]
    
        #UG = 0
    
        Q1dot = EGP0*(1.0-x3) + UG - FR - (x1+F01c/(VG*G))*Q1 + k12*Q2
        Q2dot = x1*Q1 - (k12+x2)*Q2

        S1dot = uin - S1/tmaxI
        S2dot = S1/tmaxI - S2/tmaxI

        UI = S2/tmaxI

        Idot = UI/VI - ke*I

        x1dot = -ka1*x1+kb1*I
        x2dot = -ka2*x2+kb2*I
        x3dot = -ka3*x3+kb3*I

        dstate = torch.hstack([
            Q1dot.unsqueeze(1),
            Q2dot.unsqueeze(1),
            x1dot.unsqueeze(1),
            x2dot.unsqueeze(1),
            x3dot.unsqueeze(1),
            S1dot.unsqueeze(1),
            S2dot.unsqueeze(1),
            Idot.unsqueeze(1)])
        return (dstate,G)

    Ts = 5

    def get_control(self,time,id,UG):
        #if time - self.t[id] < Controller.Ts:
        #    return self.u_prev[id]
        self.input_terms_vec0 = self.input_terms_vec0.detach()
        self.output_terms_vec0 = self.output_terms_vec0.detach()

        input_terms_vec = hypo_glucose(self.yt[id]).unsqueeze(2).repeat(1,1,self.Nemb)*self.input_terms_vec0[0,:,:].unsqueeze(0).repeat(Controller.batch_size,1,1) + target_glucose(self.yt[id]).unsqueeze(2).repeat(1,1,self.Nemb)*self.input_terms_vec0[1,:,:].unsqueeze(0).repeat(Controller.batch_size,1,1) + hyper_glucose(self.yt[id]).unsqueeze(2).repeat(1,1,self.Nemb)*self.input_terms_vec0[2,:,:].unsqueeze(0).repeat(Controller.batch_size,1,1)

        output_terms_vec = zero_dose(self.ut[id]).unsqueeze(2).repeat(1,1,self.Nemb)*self.output_terms_vec0[0,:,:].unsqueeze(0).repeat(Controller.batch_size,1,1) + low_dose(self.ut[id]).unsqueeze(2).repeat(1,1,self.Nemb)*self.output_terms_vec0[1,:,:].unsqueeze(0).repeat(Controller.batch_size,1,1) + high_dose(self.ut[id]).unsqueeze(2).repeat(1,1,self.Nemb)*self.output_terms_vec0[2,:,:].unsqueeze(0).repeat(Controller.batch_size,1,1)
    
        combined_embeds = torch.cat([
            self.vectorized_messages1[:,:-Controller.msg_end_len,:], 
            input_terms_vec, 
            #self.vectorized_messages3,
            #output_terms_vec,
            self.vectorized_messages2
        ], dim=1).detach().bfloat16()
        
        with torch.no_grad():
            outputs = self.llm(inputs_embeds=combined_embeds,use_cache=False)
        output_logits = outputs.logits[:,-1]

        zero_dose_p = output_logits.gather(1,self.output_terms_tok[0,0].repeat(Controller.batch_size,1).cuda())
        low_dose_p = output_logits.gather(1,self.output_terms_tok[1,0].repeat(Controller.batch_size,1).cuda())
        high_dose_p = output_logits.gather(1,self.output_terms_tok[2,0].repeat(Controller.batch_size,1).cuda())
    
        mf_block_norm = torch.functional.F.softmax(torch.cat([zero_dose_p,low_dose_p,high_dose_p],dim=1),dim=-1)
        agg_mu = torch.einsum('br,rx->bx', mf_block_norm, self.mf_vals.bfloat16())
        u_new = self.BW*(torch.trapz(agg_mu * self.x_vals.unsqueeze(0), self.x_vals, dim=1) / (torch.trapz(agg_mu, self.x_vals, dim=1) + 1e-8))
    
        yt_class = torch.hstack([
            hypo_glucose(self.yt[id]).mean(dim=-1).unsqueeze(1),
            target_glucose(self.yt[id]).mean(dim=-1).unsqueeze(1),
            hyper_glucose(self.yt[id]).mean(dim=-1).unsqueeze(1)
        ]).argmax(dim=-1)
    
        out_class = (self.output_terms_tok[:,0].repeat(16,1).cuda()).gather(1,yt_class.unsqueeze(1)).squeeze(1)

        loss_ce = torch.nn.CrossEntropyLoss()(output_logits,out_class)

        #if loss_ce > 1.0:
        #    u_new = u_new/loss_ce
        u_per_kg = u_new / self.BW
    
        #(k1,G) = self.ap_model(self.state[id],u_new,UG)
        #self.state[id] = self.state[id] + k1*Controller.Ts

        #state1 = state + k1*Ts*100
        #state1 = self.state[id]
        #for k in range(0,36):
        #    (k1,G) = self.ap_model(state1,u_new,UG)
        #    state1 = state1 + k1*Controller.Ts

        #self.optimizer.zero_grad()
        Gout = self.state[id][:,0]/(0.16*self.BW)
        #Gout1 = state1[:,0]/(0.16*self.BW)
        #errp = torch.sqrt(torch.clamp(Gout1 - 105/18,min=0))
        #errn = torch.clamp(105/18 - Gout1,min=0)
        Gout1 = Gout
        err1 = Gout1 - 105.0/18.0

        total_loss = (err1*err1).max() #+10.0*(u_new*u_new).mean()
        #total_loss.backward()
        #self.optimizer.step()

        self.yt[id] = torch.hstack([self.yt[id][:,self.input_toks_per_term:],Gout.unsqueeze(1).repeat(1,self.input_toks_per_term)])
        self.ut[id] = torch.hstack([self.ut[id][:,self.output_toks_per_term:],u_per_kg.unsqueeze(1)])

        print(f'id={id} u={round(u_new.mean().item())}/{round(UG.item())}/{round(self.state[id][:,5].mean().item())}/{round(self.state[id][:,7].mean().item())}, y={round(Gout.mean().item()*18)}/{round(Gout1.mean().item()*18)}, Loss={total_loss.item():.3f}/{loss_ce.item():.3f} MF: {zero_dose_p.mean().item():.3f}/{low_dose_p.mean().item():.3f}/{high_dose_p.mean().item():.3f}')
        
        self.yt[id] = self.yt[id].detach()
        self.ut[id] = self.ut[id].detach()
        self.state[id] = self.state[id].detach()
        self.t[id] = time
        self.u_prev[id] = u_per_kg

        return u_per_kg

def run(host="0.0.0.0", port=8000):
    global data1
    data1 = Controller()
    
    httpd = HTTPServer((host, port), Handler)
    print(f"Serving on http://{host}:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run()