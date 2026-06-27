mkdir data
cd data
mkdir coco
cd coco
mkdir images
cd images

wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
#wget -c http://images.cocodataset.org/zips/test2017.zip
#wget -c http://images.cocodataset.org/zips/unlabeled2017.zip

unzip train2017.zip
unzip val2017.zip
#unzip test2017.zip
#unzip unlabeled2017.zip

rm train2017.zip
rm val2017.zip
#rm test2017.zip
#rm unlabeled2017.zip 

cd ../
mkdir labels
cd labels
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
#wget -c http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
#wget -c http://images.cocodataset.org/annotations/image_info_test2017.zip
#wget -c http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

unzip annotations_trainval2017.zip
#unzip stuff_annotations_trainval2017.zip
#unzip image_info_test2017.zip
#unzip image_info_unlabeled2017.zip

rm annotations_trainval2017.zip
#rm stuff_annotations_trainval2017.zip
#rm image_info_test2017.zip
#rm image_info_unlabeled2017.zip
cd ../../../
python3 utils/coco_json_to_yolo.py -d train -j data/coco/labels/annotations/instances_train2017.json -o data/coco/labels -d train
python3 utils/coco_json_to_yolo.py -d val -j data/coco/labels/annotations/instances_val2017.json -o data/coco/labels -d val

for file in data/coco/images/train2017/*.jpg; do     new_name="data/coco/images/train2017_$(basename "$file")";     mv "$file" "$new_name"; done

for file in data/coco/images/val2017/*.jpg; do     new_name="data/coco/images/val2017_$(basename "$file")";     mv "$file" "$new_name"; done

rmdir data/coco/images/train2017
rmdir data/coco/images/val2017
rm -r  data/coco/labels/annotations

python3 utils/create_csv.py

python3 utils/clean_data.py
