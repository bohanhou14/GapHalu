class Detector:
    def __init__(self):
        self.description = 'This is a base class for detectors'

    def detect(self):
        raise NotImplementedError('detect() method is not implemented')
    
class EntityErrorDetector(Detector):
    def __init__(self):
        self.name = 'EntityErrorDetector'
        self.description = 'This detector detects entity errors in the text'

    def detect(self, generation, gold_text, prev_text, citations, short_citations):
        print('Detecting entity errors in the text')
        return image