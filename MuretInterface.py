
import GTJSONReaderMuret as muret
from CustomJson import CustomJson
from file_manager import FileManager
import numpy as np
import cv2
import random
import json 

class MuretInterface:

    @staticmethod
    def readBoundingBoxes(list_json_pathfiles, considered_classes):
        assert(type(list_json_pathfiles) is list)
        assert(considered_classes is None or type(considered_classes) is list)

        dict_bboxes = {}

        for json_pathfile in list_json_pathfiles:
            js = CustomJson()
            js.loadJson(json_pathfile)

            gtjson = muret.GTJSONReaderMuret()
            gtjson.load(js)

            #print(gtjson)

            bboxes = gtjson.getListBoundingBoxesPerClass(considered_classes)
            dict_bboxes[json_pathfile] = bboxes

        return dict_bboxes

    @staticmethod
    def getAllBoxesByRegionName(list_json_pathfiles, considered_classes = None):

        bboxes = MuretInterface.readBoundingBoxes(list_json_pathfiles = list_json_pathfiles, considered_classes = considered_classes)
        assert(type(bboxes) is dict)
        
        num_regions = 0
        dict_regions = {}
        idx = 0
        for key in bboxes:
            num_regions += len(bboxes[key])
            idx += 1

            for key_region in bboxes[key]:
                if key_region not in dict_regions:
                    dict_regions[key_region] = {}

                num_regions = len(bboxes[key][key_region])

                if len(bboxes[key][key_region]) > 0:
                    if key not in dict_regions[key_region]:
                        dict_regions[key_region][key] = []

                    dict_regions[key_region][key].append(bboxes[key][key_region])

        return dict_regions

    @staticmethod
    def applyBlurring(img, factor = 2):

        rows = img.shape[0]//factor
        cols = img.shape[1]//factor
        kernel = np.ones((rows, cols),np.float32)/(rows*cols)
        blur_img = cv2.filter2D(img,-1,kernel)
        return blur_img

    @staticmethod
    def selectRandomRegion(considered_class, dict_regions):
        candidate_images = dict_regions[considered_class]
        candidate_regions = random.choice(list(candidate_images.items()))

        selected_region_key = random.choice(candidate_regions[1][0])
        selected_json = candidate_regions[0]
        selected_img_pathfile = selected_json.replace("/JSON/", "/SRC/").replace(".json", "")
        
        selected_img = FileManager.loadImage(selected_img_pathfile, True)
        selected_patch = selected_img[selected_region_key[0]:selected_region_key[2], selected_region_key[1]:selected_region_key[3]]

        return selected_patch

        
    @staticmethod
    def generateGTImages(bboxes, shape, reduction_factor=0.3, margin=5):
        assert(len(shape) == 2)
        gt_ims = []
        
        key_region_id = 1
        for key_region in bboxes:
            gt_im = np.zeros(shape)
            
            for bbox in bboxes[key_region]:
                min_row = bbox[0]
                min_col = bbox[1]
                max_row = bbox[2]
                max_col = bbox[3]

                if reduction_factor is not None:
                    rows = max_row - min_row

                    reducted_rows = rows*(reduction_factor)
                    max_row -= int(reducted_rows)
                    min_row += int(reducted_rows)

                max_row = min(shape[0]-margin, max_row)
                min_row = max(margin, min_row)
                max_col = min(shape[1]-margin, max_col)
                min_col = max(margin, min_col)

                gt_im[min_row:max_row, min_col:max_col] = key_region_id

            gt_ims.append(gt_im)

        return gt_ims

    @staticmethod
    def splitGTImageByTypeRegion(gt_im, considered_classes):

        for considered_class in considered_classes:

            pass

        return

    @staticmethod
    def generateJSONMURET(new_bbox_regions, json_pathfile, src_filename, src_shape):
        
        with open(json_pathfile) as json_file:
            json_data = json.load(json_file)


        bbox = {}
        bbox["bounding_box"] = {"fromX":0,"toX":src_shape[0],"fromY":0,"toY":src_shape[1]}
        json_data["pages"] = []
        json_data["pages"].append(bbox)


        idx_new_bbox = 0

        if len(json_data["pages"]) == 1:
            json_data["pages"][0]['regions'] = []
            json_data["filename"] = src_filename
            json_data.pop("id")

            for key_region in new_bbox_regions:    
                for bbox_region in new_bbox_regions[key_region]:
                    bbox = {}
                    fromX = bbox_region[1]
                    toX = bbox_region[3]
                    fromY = bbox_region[0]
                    toY = bbox_region[2]
                    bbox["bounding_box"] = {"fromX":fromX,"toX":toX,"fromY":fromY,"toY":toY}
                    bbox["type"] = key_region

                    json_data["pages"][0]['regions'].append(bbox)

            idx_new_bbox += 1

        
        return json.dumps(json_data, indent = 4)



if __name__ == "__main__":

    json_dataset = "datasets/JSON/"

    list_json_pathfiles = FileManager.listFilesRecursive(json_dataset)

    bboxes = MuretInterface.readBoundingBoxes(list_json_pathfiles = list_json_pathfiles, considered_classes = None)

    #print(bboxes)

    for key in bboxes:
        print (key)
    
    print(bboxes["datasets/JSON/b-59-850/12702.JPG.json"])
