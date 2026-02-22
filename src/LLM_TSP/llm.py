#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created on 10/30/24 8:25 PM
@File:llm.py
@Author:Zhuoli Yin
@Contact: yin195@purdue.edu
'''
from openai import OpenAI
import requests
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt
import cProfile
import pstats
# from pdf2image import convert_from_path
import time

MODEL_TYPES = {
    "gpt-4o": "gpt-4o",
    "o1": "o1",
    # "qwen2.5-32b-reasoning": "Qwen/QwQ-32B",
    # "qwen2.5-32b-v":"Qwen/Qwen2.5-VL-32B-Instruct",
    # "qwen2.5-7b-v":"Qwen/Qwen2.5-VL-7B-Instruct"
}

MODEL_PRICE_INPUT = {
    'gpt-4.1-2025-04-14': 2,
    'o4-mini-2025-04-16': 8,
    'gpt-5.1-2025-11-13': 1.25
}


MODEL_PRICE_OUTPUT = {
    'gpt-4.1-2025-04-14': 1.1,
    'o4-mini-2025-04-16': 4.4,
    'gpt-5.1-2025-11-13': 10
}
class RoundRobinLLMSelector:
    """
    In case there is a rate per minute
    Use round robin to avoid server rejection
    """
    def __init__(self, llm_instances: list):
        self.llms = llm_instances
        self.counter = 0

    def get_next_llm(self):
        llm = self.llms[self.counter]
        self.counter = (self.counter + 1) % len(self.llms)
        return llm


class toy_GPT:
    def __init__(self, api_key, model_name="gpt-4o"):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
    def chat(self):
        response = self.client.responses.create(
            model= self.model_name,
            input="Return a random number between 1 and 100. Return in the format of <num> [your number] </num>"
        )
        print(response.output_text)
        return response.output_text

class GPT:
    def __init__(self, api_key, model_name="gpt-4-vision-preview"):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name

    import base64
    def vision_chat(self, fig, prior_selection, num_region, pending_coords, x_min, x_max, y_min, y_max):

        region_mapping = {
            1: 'one',
            2: 'two',
            3: 'three',  # Add more mappings as needed
        }


        # Convert PIL Image to bytes
        buf = io.BytesIO()
        fig.write_image(buf, format="png", engine='kaleido')  # Requires kaleido
        buf.seek(0)  # Move to the start of the buffer
        image = Image.open(buf).convert("RGB")

        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        # Getting the base64 string for GPT-vision
        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')

        num_region = region_mapping.get(num_region, 'one')  # 'unknown' as default value

        pending_regions = ", ".join(
            f"<coordinates> x_min={coord[0]}, x_max={coord[1]}, y_min={coord[2]}, y_max={coord[3]} </coordinates>"
            for coord in pending_coords
        )
        # Avoid selecting the same regions as follows, which are under optimizing:  <coordinates> x_min= 3,000, x_max= 4,000, y_min= 2,500, y_max=4,000 </coordinates>
        # {pending_regions}
        # Please adaptively determine the size and shape of sub-rectangles based on patterns from historical selections. <coordinates> x_min= 1,000, x_max= 2,000, y_min= 1,000, y_max=2,000 </coordinates>
        extraction_prompts = f"""You are tasked with improving an existing solution to a Traveling Salesman Problem (TSP) by selecting a sub-region where the routes can be significantly optimized. 
        Carefully consider the locations of the nodes (in red) and connected routes (in black) in the initial solution on a map. The boundary of the map is x_min={x_min-10000}, x_max={x_max+10000}, y_min={y_min-10000}, y_max={y_max+10000}.
        Please return {num_region} non-overlapping sub-rectangle(s) that you believe would most reduce total travel distance from further optimization by a downstream TSP solver.
        Analyze the problem-specific distribution to do meaningful selection. Select areas as large as you could to cover more nodes, which can bring larger improvement. Remember, if you don't see significant improvement, try selecting larger areas that cover more nodes based on your analysis of the prior selection trajectory
        Keep your output very brief as the following template. Don't tell me you cannot view or analyze the map. I don't want an excuse:
        <coordinates> x_min= 1,000, x_max= 2,000, y_min= 1,000, y_max=2,000 </coordinates> 
        \n Avoid selecting the same regions as follows, which are pending optimization:
        {pending_regions}
        
        \n Below are some previous selection trajectory. Learn from the trajectory to improve your selection capability. Please avoid selecting the same subrectangle.
        {prior_selection}
        """
        response = self.client.chat.completions.create(
            model= self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": extraction_prompts
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
           # max_tokens=100,
        )

        try:
            valid_result = response.choices[0].message.content
            print(valid_result)
            prompt_tokens = MODEL_PRICE_INPUT[self.model_name] * response.usage.prompt_tokens / 1000_000
            completion_tokens = MODEL_PRICE_OUTPUT[self.model_name] * response.usage.completion_tokens / 1000_000
            return valid_result, prompt_tokens, completion_tokens

        except requests.exceptions.RequestException as e:
            # Handles network-related errors
            print(f"Network error occurred: {e}")
            print(response.json())
        except ValueError as e:
            # Handles JSON decoding errors
            print(f"Failed to parse JSON: {e}")
            print(response.json())
        except (KeyError, IndexError) as e:
            # Handles missing or unexpected JSON structure errors
            print(f"Unexpected JSON structure: {e}")
            print(response.json())
        except Exception as e:
            # Catch-all for any other unexpected errors
            print(f"An unexpected error occurred: {e}")
            print(response.json())

# class vLLM:
#     def __init__(self, api_key=None, base_url=None, model_name="gpt-4-vision-preview"):
#         self.api_key = 'NOT_NEEDED'
#         self.client = OpenAI(base_url="http://localhost:8000/v1", api_key="NOT_NEEDED")
#         self.model_name = model_name
#
#     import base64
#     def vision_chat(self, fig, prior_selection, num_region, x_min, x_max, y_min, y_max):
#         region_mapping = {
#             1: 'one',
#             2: 'two',
#             3: 'three',  # Add more mappings as needed
#         }
#
#         buf = io.BytesIO()
#         fig.write_image(buf, format="png", engine='kaleido')  # Requires kaleido
#         buf.seek(0)  # Move to the start of the buffer
#         image = Image.open(buf).convert("RGB")
#
#         # Convert PIL Image to bytes
#         img_byte_arr = io.BytesIO()
#         image.save(img_byte_arr, format='JPEG')
#         img_byte_arr = img_byte_arr.getvalue()
#
#         # Getting the base64 string for GPT-vision
#         base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
#
#
#         # buf = io.BytesIO()
#         # fig.savefig(buf, format='JPEG')
#         # buf.seek(0)
#         # image = Image.open(buf)
#         #
#         # img_byte_arr = io.BytesIO()
#         # image.save(img_byte_arr, format='JPEG')
#         # img_byte_arr = img_byte_arr.getvalue()
#         # base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
#
#         num_region = region_mapping.get(num_region, 'one')  # 'unknown' as default value
#
#         extraction_prompts = f"""You are tasked with improving an existing solution to a Traveling Salesman Problem (TSP) by selecting a sub-region where the routes can be significantly optimized.
#         Carefully consider the locations of the nodes (in red) and connected routes (in black) in the initial solution on a map. The boundary of the map is x_min={x_min-100}, x_max={x_max+100}, y_min={y_min-100}, y_max={y_max+100}.
#         Please return {num_region} sub-rectangle(s) that you believe would most reduce total travel distance from further optimization by a downstream TSP solver. Analyze the problem-specific distribution to do meaningful selection. Please adaptively determine the size and shape of sub-rectangles based on patterns from historical selections.
#         Do not mention your thinking process but just keep your output very brief as the following template:
#         <coordinates> x_min= 1,000, x_max= 2,000, y_min= 1,000, y_max=2,000 </coordinates>;
#         Below are some previous selection trajectory. Please avoid selecting the same subrectangle.
#         {prior_selection}
#         """
#         headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {self.api_key}"
#         }
#
#         start_time = time.time()
#         chat_completion_from_base64 = self.client.chat.completions.create(
#             messages=[{
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": extraction_prompts
#                     },
#                     {
#                         "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/jpeg;base64,{base64_image}",
#                                 "detail": "high"
#                             }
#                     },
#                 ],
#             }],
#             model=MODEL_TYPES[self.model_name],
#             max_tokens=100,
#         )
#
#         print(f'{self.model_name} Latency is ', time.time() - start_time)
#         response = chat_completion_from_base64
#         try:
#             valid_result = chat_completion_from_base64.choices[0].message.content
#
#             return valid_result
#         except requests.exceptions.RequestException as e:
#             # Handles network-related errors
#             print(f"Network error occurred: {e}")
#             print(response.json())
#         except ValueError as e:
#             # Handles JSON decoding errors
#             print(f"Failed to parse JSON: {e}")
#             print(response.json())
#         except (KeyError, IndexError) as e:
#             # Handles missing or unexpected JSON structure errors
#             print(f"Unexpected JSON structure: {e}")
#             print(response.json())
#         except Exception as e:
#             # Catch-all for any other unexpected errors
#             print(f"An unexpected error occurred: {e}")
#             print(response.json())


# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
# import io
# import base64
# import time
# from PIL import Image
# import matplotlib.pyplot as plt  # assuming you're plotting something
# import torch
#
# class vLLM:
#     def __init__(self, api_key=None, base_url=None, model_name="gpt-4-vision-preview"):
#         self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#             MODEL_TYPES[model_name],
#             torch_dtype=torch.bfloat16,
#             attn_implementation="flash_attention_2",
#             device_map="auto",
#         )
#         self.processor = AutoProcessor.from_pretrained(MODEL_TYPES[model_name])
#
#     import base64
#     def vision_chat(self, fig, prior_selection, num_region, x_min, x_max, y_min, y_max):
#         region_mapping = {
#             1: 'one',
#             2: 'two',
#             3: 'three',  # Add more mappings as needed
#         }
#
#         buf = io.BytesIO()
#         fig.write_image(buf, format="png", engine='kaleido')  # Requires kaleido
#         buf.seek(0)  # Move to the start of the buffer
#         image = Image.open(buf).convert("RGB")
#
#         # Convert PIL Image to bytes
#         img_byte_arr = io.BytesIO()
#         image.save(img_byte_arr, format='JPEG')
#         img_byte_arr = img_byte_arr.getvalue()
#
#         # Getting the base64 string for GPT-vision
#         base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
#
#
#         # buf = io.BytesIO()
#         # fig.savefig(buf, format='JPEG')
#         # buf.seek(0)
#         # image = Image.open(buf)
#         #
#         # img_byte_arr = io.BytesIO()
#         # image.save(img_byte_arr, format='JPEG')
#         # img_byte_arr = img_byte_arr.getvalue()
#         # base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
#
#         num_region = region_mapping.get(num_region, 'one')  # 'unknown' as default value
#
#         extraction_prompts = f"""You are tasked with improving an existing solution to a Traveling Salesman Problem (TSP) by selecting a sub-region where the routes can be significantly optimized.
#         Carefully consider the locations of the nodes (in red) and connected routes (in black) in the initial solution on a map. The boundary of the map is x_min={x_min-100}, x_max={x_max+100}, y_min={y_min-100}, y_max={y_max+100}.
#         Please return {num_region} sub-rectangle(s) that you believe would most reduce total travel distance from further optimization by a downstream TSP solver. Analyze the problem-specific distribution to do meaningful selection. Please adaptively determine the size and shape of sub-rectangles based on patterns from historical selections.
#         Do not mention your thinking process but just keep your output very brief as the following template:
#         <coordinates> x_min= 1,000, x_max= 2,000, y_min= 1,000, y_max=2,000 </coordinates>;
#         Below are some previous selection trajectory. Please avoid selecting the same subrectangle.
#         {prior_selection}
#         """
#         # headers = {
#         #     "Content-Type": "application/json",
#         #     "Authorization": f"Bearer {self.api_key}"
#         # }
#
#         start_time = time.time()
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "image_url",
#                         "image_url": f"data:image/jpeg;base64,{base64_image}",
#                     },
#                     {"type": "text", "text": extraction_prompts},
#                 ],
#             }
#         ]
#
#         # Preparation for inference
#         text = self.processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#         image_inputs, video_inputs = process_vision_info(messages)
#         inputs = self.processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         )
#         inputs = inputs.to("cuda")
#
#         # Inference: Generation of the output
#
#         start_time = time.time()
#         generated_ids = self.model.generate(**inputs, max_new_tokens=100)
#         print('Latency is ', time.time() - start_time)
#
#         generated_ids_trimmed = [
#             out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#         ]
#         output_text = self.processor.batch_decode(
#             generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )
#         print(output_text)
#
#         response = output_text
#         try:
#             valid_result = output_text[0]
#
#             return valid_result
#         except requests.exceptions.RequestException as e:
#             # Handles network-related errors
#             print(f"Network error occurred: {e}")
#             print(response.json())
#         except ValueError as e:
#             # Handles JSON decoding errors
#             print(f"Failed to parse JSON: {e}")
#             print(response.json())
#         except (KeyError, IndexError) as e:
#             # Handles missing or unexpected JSON structure errors
#             print(f"Unexpected JSON structure: {e}")
#             print(response.json())
#         except Exception as e:
#             # Catch-all for any other unexpected errors
#             print(f"An unexpected error occurred: {e}")
#             print(response.json())
