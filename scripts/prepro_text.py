from __future__ import print_function
import sys, os
import json
import spacy

# Import spacy tokenizer
spacy_en = spacy.load('en')

# Caption data directory
data = '../data/captions/'

# Files
example_json = os.path.join(data, 'captions_val2014.json')
para_json = os.path.join(data, 'paragraphs_v1.json')
train_split = os.path.join(data, 'train_split.json')
val_split = os.path.join(data, 'val_split.json')
test_split = os.path.join(data, 'test_split.json')

# Load data
example_json = json.load(open(example_json))
para_json = json.load(open(para_json))
train_split = json.load(open(train_split))
val_split = json.load(open(val_split))
test_split = json.load(open(test_split))

# Helper function to get train/val/test split
def get_split(id):
    if id in train_split:
        return 'train'
    elif id in val_split:
        return 'val'
    elif id in test_split:
        return 'test'
    else:
        raise Exception('id not found in train/val/test')

# Tokenize the paragraphs and reformat them to a JSON file
# with the same format as the Karpathy MS COCO JSON file

def tokenize_and_reformat(para_json):
    images = []
    unique_ids = []
    for imgid, item in enumerate(para_json):

        # Log
        if imgid % 1000 == 0:
            print('{}/{}'.format(imgid, len(para_json)))

        # Extract info
        url      = item['url']                 # original url
        filename = item['url'].split('/')[-1]  # filename also is: str(item['image_id']) + '.jpg'
        id       = item['image_id']            # visual genome image id (filename)

        # Skip duplicate paragraph captions
        if id in unique_ids:
            continue
        else:
            unique_ids.append(id)

        # Extract paragraph and split into sentences
        raw_paragraph = item['paragraph']
        raw_sentences = [sent.string.strip() for sent in spacy_en(raw_paragraph).sents]   # split paragraphs

        # Tokenize paragraph and sentences
        tokenized_paragraph = [tok.text for tok in spacy_en.tokenizer(raw_paragraph) if tok.text not in [' ', '  ']]
        sentences           = [sent.capitalize() for sent in raw_sentences] 

        # Write info in coco_json format
        image = {}
        image['url']       = url
        image['filepath']  = ''                 # not relevant here (it is 'val2014' or 'train2014' for MS COCO)
        image['sentids']   = [id]               # only one ground truth paragraph 
        image['filename']  = filename
        image['imgid']     = imgid
        image['split']     = get_split(id)
        image['cocoid']    = id
        image['id']        = id 

        # This is an array for compatibility, but it always holds exactly 1 item (a dict) 
        # because we have 1 ground truth paragraph. This array hold the entire paragraph,
        # not split into sentences. 
        image['sentences'] = [{}]
        image['sentences'][0]['tokens'] = tokenized_paragraph
        image['sentences'][0]['raw']    = raw_paragraph
        image['sentences'][0]['imgid']  = imgid
        image['sentences'][0]['sentid'] = id
        image['sentences'][0]['id']     = id
        images.append(image)

    paragraph_json = {
    'images': images,
    'dataset': 'para'
    }
    
    print('Finished tokenizing paragraphs.')
    print('There are {} duplicate captions.'.format(len(para_json) - len(unique_ids)))
    print('The dataset contains {} images and annotations'.format(len(paragraph_json['images'])))

    return paragraph_json

if __name__ == '__main__':
    print('This will take a couple of minutes.')
    paragraph_json = tokenize_and_reformat(para_json)
    outfile = os.path.join(data, 'para_karpathy_format.json')
    with open(outfile, 'w') as f:
        json.dump(paragraph_json, f)

