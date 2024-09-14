import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from kolar.utils import VisionaryUtils
from kolar.prompts import build_prompt
import json

import time

class VisionaryKolar:
  
  def __init__(self) -> None:
    self.utils = VisionaryUtils()

    self.model = AutoModel.from_pretrained(
        "OpenGVLab/Mini-InternVL-Chat-2B-V1-5",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()

    self.tokenizer =  AutoTokenizer.from_pretrained("OpenGVLab/Mini-InternVL-Chat-2B-V1-5", trust_remote_code=True, use_fast=False)

  def _test_prompt(self,image_link,entity_name,question = "<image>\nDescribe the image"):
    pixel_values = self.utils.load_image(image_link).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=128, do_sample=False)
    response = self.model.chat(self.tokenizer, pixel_values, question, generation_config, history=None)
    return response

  def predict(self, image_link, entity_name:str) -> str:
    start_time = time.time()
    pixel_values = self.utils.load_image(image_link).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=128, do_sample=False)
    question = build_prompt(entity_name)
    response = self.model.chat(self.tokenizer, pixel_values, question, generation_config,history=None)
    print(f"Time taken for inference: {time.time() - start_time} seconds")
    return response
  
  def measure(self, image, entity_name:str) -> str:
    json_response = self.predict(image,entity_name)
    json_response = json_response.replace("\n","")
    if "json" in json_response:
      json_response = json_response[7:-3]
    
    try:
      response = json.loads(json_response.replace("\n","").strip())
    except:
      return "JSON decoding failed"
    return f"{response[entity_name]} {response['unit']}"

    '''
    units = False if type(response[entity_name]) is int else True
    result = str(response[entity_name])
    #print(result)
    if "unit" in response and not units:
      result += " " + str(response["unit"])

    
    return result if 20 > len(result) > 0 else ""
    '''

  def batch_predictions(self, image_links,entity_names):
    start_time = time.time()
    questions = [build_prompt(entity_name) for entity_name in entity_names]
    pixel_values = None
    num_patches_list = []
    generation_config = dict(max_new_tokens=128, do_sample=False)

    for image_link in image_links:
        pixel_value = self.utils.load_image(image_link).to(torch.bfloat16).cuda()
        
        # If pixel_values is None, initialize it with the first pixel_value
        if pixel_values is None:
            pixel_values = pixel_value
        else:
            pixel_values = torch.cat((pixel_values, pixel_value), dim=0)
            
        num_patches_list.append(pixel_value.size(0))
        

    responses = self.model.batch_chat(self.tokenizer, pixel_values,
                              num_patches_list=num_patches_list,
                              questions=questions,
                              generation_config=generation_config)
    del pixel_values
    print(f"Time taken for inference: {time.time() - start_time} seconds")
    return responses