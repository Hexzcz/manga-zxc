from ultralytics import YOLO
from src.data.core import SpeechBubble, TextCluster
import constants
from src.detection import overlap



class TextClusterDetector:
    def __init__(
        self,
        model: YOLO
    ):
        self.model = model
    

    def _detect_single(self, SBdata: SpeechBubble) -> list[TextCluster]:
        TCdata: list[TextCluster] = []

        SBx_min, SBy_min, _, _ = SBdata.position
        if SBdata.class_name == constants.FRAMELESS_CLASS_NAME:
            TCdata.append(TextCluster(
                position = SBdata.position,
                confidence = SBdata.confidence,
                bounding_image = SBdata.bounding_image,
                jp_text = None,
                tr_text = None
            ))
            return TCdata

        TCresult = self.model.predict(SBdata.bounding_image, verbose=False)[0]
        if not TCresult.boxes:
            return TCdata
        
        for box in TCresult.boxes:
            TCconf = round(box.conf.item(), 2)
            if TCconf < constants.TEXT_CLUSTER_THRESHOLD:
                continue

            TCx_min, TCy_min, TCx_max, TCy_max = map(int, box.xyxy.tolist()[0])
            relative_pos = (
                SBx_min + TCx_min, SBy_min + TCy_min, 
                SBx_min + TCx_max, SBy_min + TCy_max 
            )

            bounding_image =  SBdata.bounding_image[TCy_min:TCy_max, TCx_min:TCx_max]

            TCdata.append(TextCluster(
                position = relative_pos,
                confidence = TCconf,
                bounding_image = bounding_image,
                jp_text = None,
                tr_text = None,
            ))
        
        return TCdata


    def detect(self, SBdata: SpeechBubble | list[SpeechBubble]) -> list[TextCluster] | list[SpeechBubble]:
        if isinstance(SBdata, list):
            SBupdated_data: list[SpeechBubble] = []
            for SB in SBdata:
                TCdata = self._detect_single(SB)
                new_SB = SpeechBubble(
                    position=SB.position,
                    confidence=SB.confidence,
                    class_name=SB.class_name,
                    bounding_image=SB.bounding_image,
                    text_clusters=TCdata
                )
                SBupdated_data.append(new_SB)

            SBupdated_data = overlap.overlap(SBupdated_data)
            return SBupdated_data
        
        TCdata = self._detect_single(SBdata)
        return TCdata
        
