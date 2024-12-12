import os
import jsonlines
from tqdm import tqdm
from openai import OpenAI


class AbstractAPIExecutor:
    """
    An abstract class for API executors.
    """
    def __init__(self, model, api_key, num_payloads):
        self.model = model
        self.api_key = api_key
        self.num_payloads = num_payloads
        self.num_responses = 0
    
    def fetch_response(self):
        """
        Abstract method to fetch the response from the API.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def save_response(self, response, response_path):
        """
        Save the response to the specified file path.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    
class OpenaiAPIExecutor(AbstractAPIExecutor):
    def __init__(self, model, api_key):
        super().__init__(model, api_key, 0)
        self.client = OpenAI(api_key=api_key)
    
    def fetch_response(self, **kwargs):
        input_payloads = kwargs['input_payloads']
        self.num_payloads = len(input_payloads)
        
        print(f"[[Fetching responses]]")
        
        # Step 1: Check to cached response
        response_list = []
        if kwargs['reset'] is True:         # TODO: Fix this
            pass
        else:
            response_list = self.load_cached_response(kwargs['response_path'])
            self.num_responses = len(response_list)
            
            print(f"- Num of payloads: {self.num_payloads}")
            print(f"- Num of responses: {self.num_responses}")
            
            if self.num_responses == self.num_payloads:
                print(f"Successfully loaded the cached responses!")
                return response_list 
            elif self.num_responses > 0:
                print(f"Continuing from {self.num_responses} cached responses...")
            
        # Step 2: Fetch the responses
        for payload in tqdm(
            input_payloads[self.num_responses:],
            desc="Fetching responses",
        ):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=payload["messages"],
                    temperature=payload["temperature"],
                )
                output_response = {
                    "generated_response": response.choices[0].message.content,
                }
                self.save_response(output_response, kwargs['response_path'])
                
                response_list.append(output_response)
                
            except Exception as e:
                print(f"Error during fetching response: {e}")
            
        return response_list
    
    def save_response(self, response, response_path):
        os.makedirs(os.path.dirname(response_path), exist_ok=True)
        with jsonlines.open(response_path, mode="a") as writer:
            writer.write(response)
    
    def load_cached_response(self, response_path):
        print(f"Checking for cached responses...")
        
        if os.path.exists(response_path):
            print(f"Response already exists at '{response_path}'.")
            response_list = []
            with jsonlines.open(response_path, mode="r") as reader:
                for line in reader:
                    response_list.append(line)
            return response_list
        else:
            print(f"No cached responses found.")
            return []
        

class OllamaAPIExecutor(OpenaiAPIExecutor):
    def __init__(self, model, api_key):
        super().__init__(model, api_key)
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
    
    
class APIExecutorFactory:
    """
    A factory class to specify API executor based on the API type.
    """
    @staticmethod
    def get_api_executor(model, api_type, api_key):
        """Return an API executor based on the specified API type.
        
        Args:
            model (str): The model name.
            api_type (str): The API type.
            api_key (str): The API key.
        """
        if api_type == 'openai':
            return OpenaiAPIExecutor(model, api_key)
        if api_type == 'ollama':
            return OllamaAPIExecutor(model, api_key)
        else:
            raise ValueError(f"Unsupported API type: {api_type}.")
        
        