from enum import Enum
import base64
import os
from PIL import Image

from openai import OpenAI
from pydantic import BaseModel, create_model

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def create_response_const_types(object_list, constraint_types, target_choices, use_image=True):

    fields = {}
    for i in range(len(object_list)):
        fields[f"explanation_obj_{object_list[i]}"] = (str, ...)
        fields[f"{object_list[i]}"] = (constraint_types, ...)

    ConstraintTypes = create_model("ConstraintTypes", **fields)

    class ConstraintResponse(BaseModel):
        constraint_types: ConstraintTypes
        target_region: target_choices
        target_description: str
        if use_image:
            image_description: str
    
    return ConstraintResponse

class VLMConstraintSynthesis:
    def __init__(self, vlm_type):
        self.use_image = True
        self.client = OpenAI()
        self._vlm_type = vlm_type
        
    @property
    def agent_role_prompt(self):
        prompt_safety = """
            You are an excellent safe planning agent for dynamic tasks.
            You are given a task description and an image showing the robot and objects on a table.
            The robot is trying to slide the blue box from under the red box and to the right, without damaging other objects along the way.
        """
        return prompt_safety

    def constraint_type_prompt(self, use_image=True):
        prompt = f"""
            Each object on the table can potentially come in contact with the end-effector.
            You need to decide the safe interaction type for each object on the table from the list of constraint types.
            Here the description of the constraint types: 'no_contact' implies that there should absolutely be no contact with a certain object.
            'soft_contact' implies that you can softly interact with that object, push it softly, etc.
            'any_contact' implies that any kind of interaction including aggressive impact is allowed.
            'no_over' implies that the robot is not allowed to move over (on top of) the object.
            Some hints on how to decide on the constraint type for an object:
            If an object is soft or made of durable material, and softly pushing it or moving it without toppling it is okay, 'soft_contact' can be allowed with that object. 
            If an object is very durable, and pushing it aggressively will not damage it, 'any_contact' can be allowed with that object. 
            If an object is fragile, and contacting it might damage it, 'no_contact' should be allowed with that object.
            If an object is very sensitive like an open laptop or a bowl of food, and moving over it might be risky, 'no_over' should be constrained for that object.
            Usually objects such as cups, wine glasses, bowls, electronics, etc are considered fragile and should be 'no_contact'.
            Plastic objects such as bottles, plastic cans, tubes can be allowed 'soft_contact'.
            Soft and non-critical objects such as toys, clothing, etc are soft and can be ignored and allowed 'any_contact'.
            Provide brief explanation, for choosing a specific constraint type for an object. 
            In 'image_description' briefly describe the scene and features relevant to the task.
        """
        return prompt

    @property
    def goal_type_prompt(self):
        prompt = f"""
            You are an excellent safe planning agent for dynamic tasks.
            The robot is trying to slide the blue box from under the red box and place it in one of the two goal regions to 
            the right of the table indicated by the two yellow squares on the table.
            The square closer to the robot is bottom_goal and the square further away is top_goal.
            Choose the target region for the blue box from the two goal regions such that it is safe to slide the blue box to that target region. 
            Also provide a brief explanation for choosing that target region.
        """
        return prompt
    
    def parse_constraint_output(self, output, use_image=True):
        text = ''
        constraints = {}
        # target = output.target_region
        for obj in output.constraint_types:
            if 'explanation' in obj[0]:
                text += obj[0] + ': ' + obj[1] + '\n'
            else:
                text += obj[0] + ': ' + obj[1].value + '\n'
                constraints[obj[0]] = obj[1].value

        if use_image:
            return text, constraints
        else:
            return text, constraints
    
    def get_constraint_types(self, image_path, obj_list, constraint_types, target_choices, use_image=True):

        messages=[
            {"role": "system", "content": f"AGENT ROLE: {self.agent_role_prompt}"},
            {"role": "system", "content": f"Constraint prompt: {self.constraint_type_prompt(use_image=use_image)}"},
            {"role": "system", "content": f"Target/goal prompt: {self.goal_type_prompt}"},
        ]
        if use_image:
            base64_image = encode_image(image_path)
            messages.append(
                { 
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": "CURRENT IMAGE: Image showing the objects in the scene."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                })
            
        enum_constraint_types = Enum('enum_constraint_types', {obj: obj for i, obj in enumerate(constraint_types)}, type=str)
        enum_target_choices = Enum('target_choices', {tar: tar for i, tar in enumerate(target_choices)}, type=str)

        completion = self.client.beta.chat.completions.parse(
            model=self._vlm_type,
            messages=messages,
            response_format=create_response_const_types(obj_list, enum_constraint_types, enum_target_choices, use_image=use_image),
        )
        plan = completion.choices[0].message

        if plan.refusal: # If the model refuses to respond, you will get a refusal message
            return None

        return plan.parsed
