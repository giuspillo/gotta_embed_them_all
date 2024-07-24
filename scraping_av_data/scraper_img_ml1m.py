import requests 
from bs4 import BeautifulSoup
import pandas as pd
import urllib.request
import time
import pickle as pkl
import os

def get_items_urls():

  item_set = set(pd.read_csv('train.tsv', sep='\t', header=None)[1]).union(set(pd.read_csv('test.tsv', sep='\t', header=None)[1]))
  items = pd.DataFrame(item_set)
  items.columns = ['item_id']
  mapping = pd.read_csv('mapping_entities.tsv', sep='\t', header=None)
  mapping.columns = ['dburl', 'item_id']
  joint = items.set_index('item_id').join(mapping.set_index('item_id')).reset_index()
  return joint

def convert_url(dbpedia_url):
  dbpart = dbpedia_url.split('/')[-1]
  return 'https://en.wikipedia.org/wiki/'+dbpart

dburls = get_items_urls()
dburls.to_pickle('dburls.pkl')

for _, row in dburls.iterrows():
  
  item_id = row['item_id']
  dburl = row['dburl']
  
  try:

    wikiurl = convert_url(dburl)

    filename = 'images/' + str(item_id)+'.jpg'

    if not os.path.exists(filename):
      r = requests.get(wikiurl) 
        
      soup = BeautifulSoup(r.content, 'html5lib')

      img = soup.find('span', attrs = {'class': 'mw-default-size mw-image-border'}).find('img')['src']
      img_url = 'https:'+img.strip().split(' ')[0]
      urllib.request.urlretrieve(img_url, filename)

      print('done:' + wikiurl)
      time.sleep(6)

    else:

      print('existing: ' + wikiurl)

  except Exception as e:

    with open('log.txt', 'a') as fout:
      line = 'ERROR with movie ' + dburl + ' : ' + str(e) + '\n'
      fout.write(line)
      print(line)

      time.sleep(6)
