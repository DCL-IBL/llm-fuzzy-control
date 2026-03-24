from fuzzylogic.classes import Domain
from fuzzylogic.functions import R, S, gauss, constant
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import pdb

class FLLM():
    def __init__(self):
        self.y_domain = Domain("output", -1, 1, res=0.01)
        self.y_domain.positive = R(0,1)
        self.y_domain.negative = S(-1,0)
        self.y_domain.zero = gauss(0, 100, c_m=1)

        self.u_domain = Domain("input", -1, 1, res=0.01)
        self.u_domain.positive = R(0,1)
        self.u_domain.negative = S(-1,0)
        self.u_domain.zero = gauss(0, 100, c_m=1)

        llm_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        max_memory_map = {0: "5GiB", 1: "5GiB",2: "5GiB",}

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            device_map="auto",
            max_memory=max_memory_map,
            dtype=torch.float16,
            attn_implementation="eager"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)

        input_terms = ['positive','zero','negative']
        output_terms = ['positive','zero','negative']
        input_alternatives = len(input_terms)

        messages1 = [{"role":"system","content":"You are an assistant in making control decision."},
                     {"role":"user","content":"Output response sequence is "}]
        messages2 = [{"role":"assistant","content":"Control action sequence is "}]

        tokenized_messages1 = tokenizer.apply_chat_template(messages1, tokenize=True,
                                                            add_generation_prompt=False,
                                                            continue_final_message=False,
                                                            return_tensors="pt")
        vectorized_messages1 = llm.model.embed_tokens(tokenized_messages1[0]).unsqueeze(0)
        self.vectorized_messages1 = vectorized_messages1.repeat(input_alternatives,1,1)

        self.msg_end_len = 3

        tokenized_messages2 = tokenizer.apply_chat_template(messages2, tokenize=True, 
                                                            add_generation_prompt=False, 
                                                            continue_final_message=True,
                                                            return_tensors="pt")
        vectorized_messages2 = llm.model.embed_tokens(tokenized_messages2[0]).unsqueeze(0)
        self.vectorized_messages2 = vectorized_messages2.repeat(input_alternatives,1,1)

        input_terms_tok = tokenizer(input_terms,add_special_tokens=False)
        input_terms_tok = torch.tensor(input_terms_tok['input_ids'],dtype=torch.long).repeat(1,sequence_len)
        self.input_terms_vec0 = llm.model.embed_tokens(input_terms_tok)

        output_terms_tok = tokenizer(output_terms,add_special_tokens=False)
        self.output_terms_tok = torch.tensor(output_terms_tok['input_ids'],dtype=torch.long)

    def forward(self, y_segment):
        input_mf = [[self.y_domain.positive(y), self.y_domain.zero(y), self.y_domain.negative(y)] for y in y_segment]
        mf_vec_scale = torch.tensor(input_mf).permute(1,0).unsqueeze(2).repeat(1,1,self.input_terms_vec0.shape[2])
        input_terms_vec = mf_vec_scale.half().cuda()*self.input_terms_vec0

        combined_embeds = torch.cat([
            self.vectorized_messages1[:,:-self.msg_end_len,:], 
            input_terms_vec, 
            self.vectorized_messages1[:,-self.msg_end_len:,:],
            self.vectorized_messages2],dim=1)

        cache = DynamicCache()

        ut = []
        output_logits = torch.tensor([])
        sequence_len = len(y_segment)
        for ind in range(0,sequence_len):    
            with torch.no_grad():
                outputs = llm(inputs_embeds=combined_embeds,past_key_values=cache,use_cache=True)
            logits = outputs.logits[:,-1]
            pdb.set_trace()
            #prob = torch.functional.F.softmax(logits,dim=-1)
            output_mf = []
            for (ind,tok) in enumerate(self.output_terms_tok):
                output_mf.append(logits[ind][tok[0]])

            mf_vec_scale = torch.tensor([output_mf]).permute(1,0).unsqueeze(2).repeat(1,1,input_terms_vec0.shape[2])
            combined_embeds = mf_vec_scale.half().cuda()*input_terms_vec0
        return output_logits
            

        