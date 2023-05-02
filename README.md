# Custom BERT-based Dialogue System with ConvLab

This repository contains the implementation of a custom BERT-based dialogue system using the ConvLab-2 framework for the KVRET dataset. The main focus of this project is the development of a custom NLU component, while the other components, such as DST, policy, and NLG, are based on pre-built models provided by ConvLab-2.

## Installation

To use this code, first clone the repository and install the necessary dependencies. You'll need Python 3.7 or later and the following packages:

```bash
pip install torch
pip install transformers
pip install convlab2
pip install sklearn
```
## Usage

After installing the required packages, you'll need to provide paths to the pretrained models and tokenizers for intent detection and slot filling. The code in this repository assumes that you have already trained these models. If you haven't, you'll need to train them first and save the weights, tokenizers, and label encoders.

The main components of the dialogue system are created as follows:

```python
from convlab2.dialog_agent import PipelineAgent
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.nlg.template.multiwoz import TemplateNLG
from custom_bert_nlu import CustomBERTNLU

# Provide paths to your trained models, tokenizers, and label encoders
intent_model_path = "/path/to/intent/model/"
slot_model_path = "/path/to/slot/model/"
tokenizer_path = "/path/to/tokenizer/"
intent_label_encoder_path = "/path/to/intent/label/encoder.pkl"
slot_label_encoder_path = "/path/to/slot/label/encoder.pkl"

# Create an instance of your custom NLU component
nlu = CustomBERTNLU(
    intent_model_path, 
    slot_model_path, 
    tokenizer_path, 
    intent_label_encoder_path, 
    slot_label_encoder_path
)

# Set up the dialogue system components
dst = RuleDST()
policy = RulePolicy()
nlg = TemplateNLG(is_user=False)
agent = PipelineAgent(nlu, dst, policy, nlg, name='sys')

# Now you can use the agent to process user inputs
user_input = "Can you make a system please?"
agent_response = agent.response(user_input)
print(agent_response)

With the dialogue system set up, you can interact with it by providing user inputs and receiving the system's responses.
```
## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project is built upon the ConvLab framework developed by the Tsinghua University Conversational AI Group.
The BERT models and tokenizers used in this project are based on the Transformers library by Hugging Face.
