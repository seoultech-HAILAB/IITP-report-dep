import os
import datasets


class AbstractDataLoader:
    """
    An abstract class for data loaders.
    """
    def __init__(self, dataset_name, dataset_size=100):
        self.dataset_name = dataset_name
        self.dataset_size = dataset_size
    
    def load_dataset(self):
        """
        Abstract method to load the dataset.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def save_dataset(self, dataset, dataset_path):
        """
        Save the dataset to the specified file path.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    

class HuggingFaceDataLoader(AbstractDataLoader):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
    
    def load_dataset(self, **kwargs):
        print("[[Loading dataset]]")
            
        # Step 1. Check to cached dataset
        if kwargs['reset'] is True:         # TODO: Fix this
            pass
        else:
            input_dataset = self.load_cached_dataset(kwargs['dataset_path'])
            if len(input_dataset) > 0:      # TODO: Fix this
                print(f"Successfully loaded the cached dataset!")
                return input_dataset            
            
        # Step 2. Download the dataset
        print(f"Downloading the dataset from HuggingFace...")
        downloaded_dataset = datasets.load_dataset(self.dataset_name, split=kwargs['split'])
        
        # Step 3. Save the dataset to the specified file path
        self.save_dataset(downloaded_dataset, kwargs['dataset_path'])
        print(f"Saved dataset to '{kwargs['dataset_path']}'.")
        return datasets.load_from_disk(kwargs['dataset_path'])
    
    def save_dataset(self, dataset, dataset_path):
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        dataset.save_to_disk(dataset_path)
    
    def load_cached_dataset(self, dataset_path):
        print(f"Checking for cached '{self.dataset_name}' dataset...")
        
        if os.path.exists(dataset_path):
            print(f"Dataset already exists at '{dataset_path}'.")
            return datasets.load_from_disk(dataset_path)
        else:
            print(f"Dataset does not exist at '{dataset_path}'.")
            return []
    

class DataLoaderFactory:
    """
    A factory class to specify the data loader based on the source.
    """
    @staticmethod
    def get_data_loader(source, dataset_name):
        """
        Returns an instance of the data loader based on the specified source.
        """
        if source == "huggingface":
            return HuggingFaceDataLoader(dataset_name)
        else:
            raise ValueError(f"Unknown data source: {source}")