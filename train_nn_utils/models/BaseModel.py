
class BaseModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    def _load_model(self):
        pass
    
    def get_model(self):
        self._load_model()
        return self.model