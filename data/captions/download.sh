# Get train/val/test splits
wget https://cs.stanford.edu/people/ranjaykrishna/im2p/train_split.json --no-check-certificate
wget https://cs.stanford.edu/people/ranjaykrishna/im2p/val_split.json --no-check-certificate
wget https://cs.stanford.edu/people/ranjaykrishna/im2p/test_split.json --no-check-certificate

# Get paragraph captions
wget http://visualgenome.org/static/data/dataset/paragraphs_v1.json.zip
unzip paragraphs_v1.json.zip
rm *flickr* *.zip

# Get COCO validation JSON as example
wget https://github.com/tylin/coco-caption/raw/master/annotations/captions_val2014.json

