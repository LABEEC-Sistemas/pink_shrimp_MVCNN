import cv2 as cv
from pathlib import Path

class IndividuoCamarao:
    def __init__(self, somito_6_path:str, carapaca_path:str, suco_adrostral_path:str)->None:
        self._somito_6_image = None
        self.somito_6_path:Path = Path(somito_6_path)
        self._carapaca_image = None
        self.carapaca_path:Path = Path(carapaca_path)
        self._suco_adrostral_image = None
        self.suco_adrostral_path:Path = Path(suco_adrostral_path)
    
    # @property é um recurso da orientação a objetos do python que permite que um método seja acessado como se fosse um atributo.
    @property
    def somito_6_image(self):

        if self._somito_6_image is None:
            self._somito_6_image = cv.imread(self.somito_6_path)

        return self._somito_6_image
    
    @property
    def carapaca_image(self):

        if self._carapaca_image is None:
            self._carapaca_image = cv.imread(self.carapaca_path)

        return self._carapaca_image
    
    @property
    def suco_adrostral_image(self):

        if self._suco_adrostral_image is None:
            self._suco_adrostral_image = cv.imread(self.suco_adrostral_path)

        return self._suco_adrostral_image

    @property
    def name(self):
        return self.somito_6_path.parent.name

    @property
    def especie(self):
        return self.somito_6_path.parent.parent.name

    def get_images(self):
         return {
            "6_somito": cv.cvtColor(self.somito_6_image, cv.COLOR_BGR2RGB),
            "carapaca": cv.cvtColor(self.carapaca_image, cv.COLOR_BGR2RGB),
            "suco_adrostral": cv.cvtColor(self.suco_adrostral_image, cv.COLOR_BGR2RGB)
        }