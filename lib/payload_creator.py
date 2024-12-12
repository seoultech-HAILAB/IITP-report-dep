import os
import jsonlines
from tqdm import tqdm


class AbstractPayloadCreator:
    """
    An abstract class for payload creators.
    """
    def __init__(self, temperature, num_examples, system_prompt_path):
        self.temperature = temperature
        self.num_examples = num_examples
        self.num_payloads = 0
        if system_prompt_path:
            self.system_prompt = self._get_prompt_txt(system_prompt_path)
    
    def create_payload(self, **kwargs):
        """
        Abstract method to create a payload for an API request.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def save_payload(self, api_request_list):
        """
        Save the payloads to the specified file path.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _get_prompt_txt(self, system_prompt_path):
        """Returns the system prompt text from the specified file path.

        Args:
            system_prompt_path (str): The file path to the system prompt text.

        Returns:
            str: The system prompt text.
        """
        system_prompt = ""
        if system_prompt_path and os.path.isfile(system_prompt_path):
            with open(system_prompt_path, "r") as f:
                system_prompt = f.read()
        return system_prompt


class QuestionGenerationPayloadCreator(AbstractPayloadCreator):
    def __init__(self, temperature, system_prompt_path):
        super().__init__(temperature, 0, system_prompt_path)

    def create_payload(self, **kwargs):
        input_dataset = kwargs['input_dataset']
        self.num_examples = len(input_dataset)
        
        print(f"[[Creating payloads]]")
               
        # Step 1: Check to cached payloads
        api_request_list = []
        if kwargs['reset'] is True:         # TODO: Fix this
            pass
        else:
            api_request_list = self.load_cached_payload(kwargs['payload_path'])
            self.num_payloads = len(api_request_list)
            
            print(f"- Num of examples: {self.num_examples}")
            print(f"- Num of payloads: {self.num_payloads}")
            
            if self.num_payloads == self.num_examples:
                print(f"Successfully loaded the cached payloads!")
                return api_request_list
            elif self.num_payloads > 0:
                print(f"Continuing from {self.num_payloads} cached payloads...")
            
        # Step 2: Create the payloads
        for input_data in tqdm(
            input_dataset.select(range(self.num_payloads, self.num_examples)),
            desc="Creating payloads",
        ):
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user", 
                    "content": f"Paragraph: {input_data['paragraph']}\nAnswer: {input_data['answer']}"
                },
            ]
            payload = {
                "messages": messages,
                "temperature": self.temperature,
                "ground_truth": input_data["question"],
            }
            self.save_payload(payload, kwargs['payload_path'])

            api_request_list.append(payload)
            
        return api_request_list
    
    def save_payload(self, payload, payload_path):
        os.makedirs(os.path.dirname(payload_path), exist_ok=True)
        with jsonlines.open(payload_path, mode="a") as writer:
            writer.write(payload)

    def load_cached_payload(self, payloads_path):
        print(f"Checking for cached payloads...")
        
        if os.path.exists(payloads_path):
            print(f"Payloads already exists at '{payloads_path}'.")
            api_request_list = []
            with jsonlines.open(payloads_path, mode="r") as reader:
                for line in reader:
                    api_request_list.append(line)
            return api_request_list
        else:
            print(f"No cached payloads found.")
            return []


class PayloadCreatorFactory:
    """
    A factory class to specify payload creator based on the type of task.
    """
    @staticmethod
    def get_payload_creator(task_type, temperature, system_prompt_path):
        """Returns an instance of the payload creator based on the specified task type.
        
        Args:
            task_type (str): The type of task for which the payload creator is needed.
        """
        if task_type == "QG":
            return QuestionGenerationPayloadCreator(temperature, system_prompt_path)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        